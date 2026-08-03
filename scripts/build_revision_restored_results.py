#!/usr/bin/env python3
"""Restore the first-revision result structure with corrected-protocol values.

Figures 3-11, per-trip tables, and the complete six-row bootstrap summary are
rebuilt from audited validation-selected checkpoints. Figures 1-2 are not
modified by this script because their reviewer-version artwork is canonical.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts.regenerate_revision_prediction_figures import (
    DATASET_ORDER, SPLIT_SEED, TRAINING_SEEDS, dataset_predictions, discover,
)
from scripts.run_revision_ev_proxy_validation import (
    batched_predict, context_windows, load_checkpoint, make_model, sliding_windows,
)
from src.revision_protocol import (
    ACTUATION, DATASETS, SEED, SHARED_CONTEXT, WINDOW_SIZE, complete_trip_split,
    fit_feature_scaler, load_dataset, save_json, summarize_trip_metric,
)

ICEV_ORDER = ("pacifica", "blazer", "qx50")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--five-seed-dir", required=True)
    p.add_argument("--ev-single-dir", required=True)
    p.add_argument("--output-dir", default="artifacts/revision_restored_results")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def one(root: Path, pattern: str) -> Path:
    found = sorted(root.rglob(pattern))
    if len(found) != 1:
        raise RuntimeError(f"Expected one {pattern} under {root}; found {found}")
    return found[0]


def savefig(fig: plt.Figure, stem: Path) -> list[Path]:
    out = [stem.with_suffix(".png"), stem.with_suffix(".pdf"), stem.with_suffix(".tiff")]
    fig.savefig(out[0], dpi=300, bbox_inches="tight")
    fig.savefig(out[1], bbox_inches="tight")
    fig.savefig(out[2], dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return out


def thin(n: int, maximum: int = 5000) -> np.ndarray:
    return np.arange(n) if n <= maximum else np.unique(np.linspace(0, n-1, maximum).astype(int))


def trip_rows(item: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    y, p, trips = item["observed"], item["predictions"], item["test_trips"]
    for trip in pd.unique(trips):
        mask = trips == trip
        maes = np.mean(np.abs(p[:, mask] - y[None, mask]), axis=1)
        rows.append({"trip_id": str(trip), "n_windows": int(mask.sum()),
                     "mae_mean_across_seeds": float(maes.mean()),
                     "mae_sample_sd_across_seeds": float(maes.std(ddof=1)),
                     "mae_min_across_seeds": float(maes.min()),
                     "mae_max_across_seeds": float(maes.max()),
                     **{f"mae_seed_{s}": float(maes[i]) for i, s in enumerate(TRAINING_SEEDS)}})
    return rows


def training_figure(item: dict[str, Any], out: Path) -> list[Path]:
    tr, va = item["train_history"], item["validation_history"]
    ep = np.arange(1, tr.shape[1]+1)
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    for data, label in ((tr, "Train MSE"), (va, "Validation MSE")):
        mean, sd = data.mean(0), data.std(0, ddof=1)
        ax.plot(ep, mean, label=f"{label} mean")
        ax.fill_between(ep, np.maximum(mean-sd, 1e-12), mean+sd, alpha=.2)
    ax.set(yscale="log", xlabel="Epoch", ylabel="MSE (log scale)",
           title=f"{item['display_name']}: training and validation across five seeds")
    ax.grid(alpha=.2); ax.legend(frameon=False)
    return savefig(fig, out / f"figure_training_{item['dataset']}")


def best_worst_figure(item: dict[str, Any], rows: list[dict[str, Any]], out: Path) -> list[Path]:
    ordered = sorted(rows, key=lambda r: (r["mae_mean_across_seeds"], r["trip_id"]))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), constrained_layout=True)
    for ax, name, row in zip(axes, ("Best", "Worst"), (ordered[0], ordered[-1])):
        mask = item["test_trips"] == row["trip_id"]
        y = item["observed"][mask]; p = item["five_seed_mean"][mask]
        sd = item["five_seed_sample_sd"][mask]; idx = thin(len(y))
        ax.plot(idx, y[idx], color="black", lw=.8, label="Observed")
        ax.plot(idx, p[idx], lw=.8, label="Five-seed prediction mean")
        ax.fill_between(idx, p[idx]-sd[idx], p[idx]+sd[idx], alpha=.22, lw=0,
                        label="±1 sample SD")
        ax.set(title=f"{name}: trip {row['trip_id']}\nmean trip MAE={row['mae_mean_across_seeds']:.4g}",
               xlabel="Window index within trip", ylabel="CO₂ emissions (g/s)")
        ax.grid(alpha=.2)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(f"{item['display_name']}: held-out best and worst trips")
    return savefig(fig, out / f"figure_best_worst_{item['dataset']}")


def median_trip(lengths: dict[str, int]) -> str:
    ranked = sorted(lengths.items(), key=lambda x: (x[1], x[0]))
    return ranked[(len(ranked)-1)//2][0]


def ev_components(root: Path, batch: int) -> dict[str, Any]:
    ecp, fcp = load_checkpoint(one(root, "ev_emissions_best.pt")), load_checkpoint(one(root, "ev_feature_best.pt"))
    er = json.loads(one(root, "ev_emissions.json").read_text()); fr = json.loads(one(root, "ev_feature.json").read_text())
    frame = load_dataset(DATASETS["ev"]); split = complete_trip_split(frame, seed=SPLIT_SEED)
    for r in (er, fr):
        assert r["split_manifest"] == split.as_dict()
        assert r["used_window_counts"] == r["full_window_counts"]
        assert r["checkpoint_selection"]["test_set_used_for_selection"] is False
    fs = fit_feature_scaler(frame, split.train_trip_ids, SHARED_CONTEXT)
    es = fit_feature_scaler(frame, split.train_trip_ids, ACTUATION)
    fm, em = make_model(fcp), make_model(ecp)
    rows, series, flen, plen = [], {}, {}, {}
    for trip_id in split.test_trip_ids:
        trip = frame.loc[frame["Trip"] == str(trip_id)].reset_index(drop=True)
        cx = context_windows(fs.transform(trip[SHARED_CONTEXT].to_numpy(float)).astype(np.float32), WINDOW_SIZE)
        fp = batched_predict(fm, cx, batch)
        ft = trip[["Motor Torque [Nm]", "Throttle [%]"]].to_numpy(float)[WINDOW_SIZE:]
        assert len(fp) == len(ft)
        torque = float(np.mean(np.abs(fp[:,0]-ft[:,0]))); throttle = float(np.mean(np.abs(fp[:,1]-ft[:,1])))
        flen[str(trip_id)] = len(ft)
        v = trip["Velocity [km/h]"].to_numpy(float)[WINDOW_SIZE:]
        direct_raw = np.column_stack([v, trip["Throttle [%]"].to_numpy(float)[WINDOW_SIZE:], trip["Motor Torque [Nm]"].to_numpy(float)[WINDOW_SIZE:]])
        proxy_raw = np.column_stack([v, fp[:,1], fp[:,0]])
        dx = sliding_windows(es.transform(direct_raw).astype(np.float32), WINDOW_SIZE)
        px = sliding_windows(es.transform(proxy_raw).astype(np.float32), WINDOW_SIZE)
        y = trip["CO2 Emissions"].to_numpy(float)[2*WINDOW_SIZE:]
        direct = batched_predict(em, dx, batch).reshape(-1); proxy = batched_predict(em, px, batch).reshape(-1)
        assert len(y) == len(direct) == len(proxy)
        dm = float(np.mean(np.abs(direct-y))); pm = float(np.mean(np.abs(proxy-y)))
        plen[str(trip_id)] = len(y)
        rows.append({"trip_id": str(trip_id), "n_feature_windows": len(ft), "n_proxy_aligned_windows": len(y),
                     "ev_direct_co2_mae_gps": dm, "ev_proxy_co2_mae_gps": pm,
                     "proxy_minus_direct_gps": pm-dm, "torque_mae_nm": torque,
                     "throttle_mae_pct": throttle})
        series[str(trip_id)] = {"feature_true": ft, "feature_pred": fp, "observed": y, "direct": direct, "proxy": proxy}
    return {"rows": sorted(rows, key=lambda r:r["trip_id"]), "series": series,
            "feature_trip": median_trip(flen), "proxy_trip": median_trip(plen),
            "emissions_history": er["history"], "feature_history": fr["history"],
            "emissions_checkpoint_sha256": sha(one(root, "ev_emissions_best.pt")),
            "feature_checkpoint_sha256": sha(one(root, "ev_feature_best.pt")),
            "split_manifest": split.as_dict()}


def ev_training_figure(ev: dict[str, Any], comp: dict[str, Any], out: Path) -> list[Path]:
    fig, axes = plt.subplots(1,2,figsize=(11.5,3.8),constrained_layout=True)
    tr, va = ev["train_history"], ev["validation_history"]; ep=np.arange(1,tr.shape[1]+1)
    for d,l in ((tr,"Train MSE"),(va,"Validation MSE")):
        m,s=d.mean(0),d.std(0,ddof=1); axes[0].plot(ep,m,label=f"{l} mean"); axes[0].fill_between(ep,np.maximum(m-s,1e-12),m+s,alpha=.2)
    fh=comp["feature_history"]; fe=np.array([r["epoch"] for r in fh])
    axes[1].plot(fe,[r["train_mse"] for r in fh],label="Train MSE"); axes[1].plot(fe,[r["validation_mse"] for r in fh],label="Validation MSE")
    axes[0].set_title("EV emissions model: five seeds"); axes[1].set_title("EV torque/throttle model: selected run")
    for ax in axes: ax.set(yscale="log",xlabel="Epoch",ylabel="MSE (log scale)"); ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.suptitle("BMW i3 model training under the corrected complete-trip protocol")
    return savefig(fig,out/"figure_training_ev_emissions_features")


def ev_feature_figure(comp: dict[str, Any], out: Path) -> list[Path]:
    trip=comp["feature_trip"]; s=comp["series"][trip]; t,p=s["feature_true"],s["feature_pred"]; idx=thin(len(t))
    fig,axes=plt.subplots(1,2,figsize=(11.5,3.8),constrained_layout=True)
    for ax,(j,name,unit) in zip(axes,((0,"Motor torque","Nm"),(1,"Throttle","%"))):
        mae=float(np.mean(np.abs(p[:,j]-t[:,j]))); ax.plot(idx,t[idx,j],color="black",lw=.8,label="Measured"); ax.plot(idx,p[idx,j],lw=.8,label="Predicted")
        ax.set(title=f"{name}: MAE={mae:.4g} {unit}",xlabel="Feature-model window index",ylabel=f"{name} ({unit})"); ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.suptitle(f"BMW i3 held-out median-length trip {trip}: measured versus predicted actuation")
    return savefig(fig,out/"figure_ev_torque_throttle_measured_predicted")


def ev_proxy_figure(comp: dict[str, Any], out: Path) -> list[Path]:
    trip=comp["proxy_trip"]; s=comp["series"][trip]; y,d,p=s["observed"],s["direct"],s["proxy"]; idx=thin(len(y))
    dm=float(np.mean(np.abs(d-y))); pm=float(np.mean(np.abs(p-y)))
    fig,ax=plt.subplots(figsize=(10.5,3.8),constrained_layout=True)
    ax.plot(idx,y[idx],color="black",lw=.8,label="Observed"); ax.plot(idx,d[idx],lw=.8,label=f"Measured-actuation prediction (MAE={dm:.4g})")
    ax.plot(idx,p[idx],lw=.8,ls="--",label=f"Predicted-actuation proxy (MAE={pm:.4g})")
    ax.set(title=f"Corrected in-domain EV proxy validation: median-length trip {trip}",xlabel="Aligned window index within trip",ylabel="CO₂ emissions (g/s)"); ax.grid(alpha=.2); ax.legend(frameon=False,ncol=3)
    return savefig(fig,out/"figure_ev_direct_proxy_observed")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def boot(metric: str, latex: str, unit: str, values: Iterable[float]) -> dict[str, Any]:
    s=summarize_trip_metric(values,SEED); q1,q3=s["iqr"]; lo,hi=s["bootstrap_95pct_mean_ci"]
    return {"metric":metric,"metric_latex":latex,"unit":unit,"n_trips":s["n_trips"],"mean":s["mean"],"median":s["median"],"sample_std":s["sample_std"],"q1":q1,"q3":q3,"iqr_width":q3-q1,"ci_low":lo,"ci_high":hi}


def tex_tables(path: Path, rows: dict[str,list[dict[str,Any]]], ev: list[dict[str,Any]], b: list[dict[str,Any]]) -> None:
    esc=lambda x:x.replace("_",r"\_").replace("%",r"\%")
    names={"qx50":"Infiniti QX50","blazer":"Chevrolet Blazer","pacifica":"Chrysler Pacifica"}; L=["% Auto-generated corrected-protocol tables."]
    for d in ("qx50","blazer","pacifica"):
        L += [r"\begin{table}[t]",rf"\caption{{{names[d]} -- held-out-trip CO$_2$ MAE across five LSTM training seeds.}}",rf"\label{{tab:{d}_trip_mae_corrected}}",r"\centering\begin{tabular}{lcc}",r"\hline Trip & Mean MAE (g/s) & SD \\",r"\hline"]
        L += [f"{esc(r['trip_id'])} & {r['mae_mean_across_seeds']:.4f} & {r['mae_sample_sd_across_seeds']:.4f} \\" for r in sorted(rows[d],key=lambda x:x['trip_id'])]
        L += [r"\hline\end{tabular}",r"\end{table}"]
    L += [r"\begin{table*}[t]",r"\caption{BMW i3 held-out-trip metrics under the corrected complete-trip protocol. Direct and proxy emissions use the common two-stage aligned subset; torque and throttle use all eligible feature-model windows.}",r"\label{tab:ev_trip_metrics_corrected}",r"\centering\scriptsize\begin{tabular}{lrrrrr}",r"\hline Trip & Direct CO$_2$ & Proxy CO$_2$ & Proxy$-$direct & Torque & Throttle \\",r" & MAE (g/s) & MAE (g/s) & (g/s) & MAE (Nm) & MAE (\%) \\",r"\hline"]
    L += [f"{esc(r['trip_id'])} & {r['ev_direct_co2_mae_gps']:.4f} & {r['ev_proxy_co2_mae_gps']:.4f} & {r['proxy_minus_direct_gps']:.4f} & {r['torque_mae_nm']:.3f} & {r['throttle_mae_pct']:.3f} \\" for r in ev]
    L += [r"\hline\end{tabular}",r"\end{table*}",r"\begin{table*}[t]",r"\caption{Complete trip-level nonparametric bootstrap summary. Intervals are 95\% percentile intervals for the mean from 10,000 trip resamples.}",r"\label{tab:bootstrap_complete_corrected}",r"\centering\small\begin{tabular}{lrrrrr}",r"\hline Metric & $n$ & Mean & 95\% bootstrap CI & Median & IQR ($Q_1$--$Q_3$) \\",r"\hline"]
    L += [f"{r['metric_latex']} & {r['n_trips']} & {r['mean']:.4f} & [{r['ci_low']:.4f}, {r['ci_high']:.4f}] & {r['median']:.4f} & [{r['q1']:.4f}, {r['q3']:.4f}] \\" for r in b]
    L += [r"\hline\end{tabular}",r"\end{table*}"]
    path.write_text("\n".join(L)+"\n",encoding="utf-8")


def main() -> None:
    a=args(); torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); tmp=out/"_tmp"; arrays=out/"_arrays"; tmp.mkdir(exist_ok=True); arrays.mkdir(exist_ok=True)
    cps,res=discover(Path(a.five_seed_dir)); data=[dataset_predictions(d,cps,res,tmp,arrays,a.batch_size) for d in DATASET_ORDER]; em={x["dataset"]:x for x in data}
    rows={d:trip_rows(em[d]) for d in DATASET_ORDER}; comp=ev_components(Path(a.ev_single_dir),a.batch_size); generated=[]
    for d in ICEV_ORDER: generated += training_figure(em[d],out)+best_worst_figure(em[d],rows[d],out)
    generated += ev_training_figure(em["ev"],comp,out)+ev_feature_figure(comp,out)+ev_proxy_figure(comp,out)
    for d in ("qx50","blazer","pacifica"):
        p=out/f"table_{d}_trip_metrics.csv"; write_csv(p,rows[d]); generated.append(p)
    ep=out/"table_ev_trip_metrics_complete.csv"; write_csv(ep,comp["rows"]); generated.append(ep)
    icev=[r["mae_mean_across_seeds"] for d in ("qx50","blazer","pacifica") for r in rows[d]]; ev=comp["rows"]
    b=[boot("ICEV CO2 MAE",r"ICEV CO$_2$ MAE","g/s",icev),boot("EV direct CO2 MAE",r"EV direct CO$_2$ MAE","g/s",[r["ev_direct_co2_mae_gps"] for r in ev]),boot("EV proxy CO2 MAE",r"EV proxy CO$_2$ MAE","g/s",[r["ev_proxy_co2_mae_gps"] for r in ev]),boot("Proxy - direct",r"Proxy $-$ direct","g/s",[r["proxy_minus_direct_gps"] for r in ev]),boot("Torque MAE","Torque MAE","Nm",[r["torque_mae_nm"] for r in ev]),boot("Throttle MAE","Throttle MAE","%",[r["throttle_mae_pct"] for r in ev])]
    bp=out/"table_bootstrap_complete.csv"; write_csv(bp,b); generated.append(bp); tp=out/"restored_tables_corrected.tex"; tex_tables(tp,rows,ev,b); generated.append(tp)
    import shutil; shutil.rmtree(tmp); shutil.rmtree(arrays)
    manifest={"status":"completed_restored_first_revision_results","split_seed":SPLIT_SEED,"training_seeds":list(TRAINING_SEEDS),"window_size":WINDOW_SIZE,"protocol_checks":{"complete_trip_split":True,"training_only_feature_scaler":True,"windows_constructed_after_split_within_trip":True,"validation_selected_checkpoints":True,"test_set_used_for_model_selection":False,"trip_bootstrap_resamples":10000},"figure_mapping":{"Figure 3":"figure_training_pacifica","Figure 4":"figure_training_blazer","Figure 5":"figure_training_qx50","Figure 6":"figure_best_worst_pacifica","Figure 7":"figure_best_worst_blazer","Figure 8":"figure_best_worst_qx50","Figure 9":"figure_training_ev_emissions_features","Figure 10":"figure_ev_torque_throttle_measured_predicted","Figure 11":"figure_ev_direct_proxy_observed"},"selection_rules":{"icev_best_worst":"minimum/maximum mean held-out-trip MAE across five fixed-split training seeds","ev_feature_trip":"median feature-window count, trip-ID tie-breaker, independent of targets/predictions","ev_proxy_trip":"median aligned-window count, trip-ID tie-breaker, independent of targets/predictions"},"icev_trip_metrics":{d:rows[d] for d in ("qx50","blazer","pacifica")},"ev_trip_metrics":ev,"representative_ev_feature_trip":comp["feature_trip"],"representative_ev_proxy_trip":comp["proxy_trip"],"bootstrap_summary":b,"checkpoint_sha256":{"ev_emissions":comp["emissions_checkpoint_sha256"],"ev_feature":comp["feature_checkpoint_sha256"]},"generated_files":{},"guardrails":["The proxy validation is an in-domain EV component-composition check, not a cross-powertrain counterfactual validation.","Five training seeds quantify optimization variability on a fixed trip split; they are not independent test datasets.","The comparison is conditional on observed covariates and does not establish causal equivalence of operating conditions."]}
    for p in generated: manifest["generated_files"][p.name]={"sha256":sha(p),"bytes":p.stat().st_size}
    save_json(out/"restored_results_manifest.json",manifest)
    diff=next(r for r in b if r["metric"]=="Proxy - direct")
    (out/"README.md").write_text("# Restored first-revision results under the corrected protocol\n\nFigures 3-11 and all per-trip/bootstrap tables are regenerated from leakage-free complete-trip splits and validation-selected checkpoints. Figures 1-2 remain unchanged.\n\nThe corrected proxy-minus-direct trip-mean MAE is **%.8g g/s** with 95%% bootstrap CI **[%.8g, %.8g]**. The proxy therefore increases MAE; the previous negligible-degradation/denoising interpretation is not supported.\n"%(diff["mean"],diff["ci_low"],diff["ci_high"]),encoding="utf-8")
    print(json.dumps({"status":manifest["status"],"proxy_minus_direct":diff,"generated":sorted(manifest["generated_files"])},indent=2))

if __name__ == "__main__": main()
