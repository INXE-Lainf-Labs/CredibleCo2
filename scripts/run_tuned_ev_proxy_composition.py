#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from src.models.LSTM import MultipleLayerLSTM

SPLIT_SEED=20260801
SHA='8ccb1489d4d041e5688e1aa808921c1f7694fb6968521f77b619bee8cafb8262'
EM=['Velocity [km/h]','Throttle [%]','Motor Torque [Nm]']
CTX=['Velocity [km/h]','Ambient Temperature [°C]','Cabin Temperature Sensor [°C]','Longitudinal Acceleration [m/s^2]']
ACT=['Motor Torque [Nm]','Throttle [%]']

def parse():
 p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--emissions-checkpoint',required=True); p.add_argument('--feature-checkpoint',required=True); p.add_argument('--training-seed',type=int,required=True); p.add_argument('--output-dir',default='artifacts/tuned_ev_proxy_composition'); p.add_argument('--batch-size',type=int,default=2048); return p.parse_args()
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def load(path,task,training_seed):
 c=torch.load(path,map_location='cpu',weights_only=False); q=c['payload']
 ok=(q['record_type']=='lstm_tuning_run' and q['stage']=='final' and q['task']==task and q['dataset']=='ev' and q['split_seed']==SPLIT_SEED and q['training_seed']==training_seed and not q['test_set_used_for_selection'] and q['checkpoint_selection']['restored_before_test'] and q['used_window_counts']==q['full_window_counts'])
 if not ok: raise RuntimeError('invalid checkpoint '+path)
 return c
def mdl(c):
 q=c['payload']; g=q['configuration']; m=MultipleLayerLSTM(len(q['feature_columns']),g['hidden_dim'],len(q['target_columns']),g['num_blocks']); m.load_state_dict(c['model_state_dict']); return m.eval()
def scale(x,c): return (x*np.asarray(c['feature_scaler_scale'])+np.asarray(c['feature_scaler_min'])).astype(np.float32)
def inv(y,c): return y.astype(float) if c['payload']['configuration']['target_mode']=='raw' else y.astype(float)*np.asarray(c['target_std'])+np.asarray(c['target_mean'])
def win(x,w):
 z=np.lib.stride_tricks.sliding_window_view(x,w,axis=0); return z[:-1].transpose(0,2,1).copy().astype(np.float32)
def pred(m,x,b):
 o=[]
 with torch.inference_mode():
  for i in range(0,len(x),b): o.append(m(torch.from_numpy(x[i:i+b])).numpy())
 return np.concatenate(o)
def met(y,p):
 e=p-y; den=np.square(y-y.mean()).sum(); return {'mae':float(np.abs(e).mean()),'rmse':float(np.sqrt(np.square(e).mean())),'r2':float(1-np.square(e).sum()/den)}
def summ(a,seed):
 a=np.asarray(a,float); r=np.random.default_rng(seed); bm=a[r.integers(0,len(a),(10000,len(a)))].mean(1); return {'n_trips':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'sample_std':float(a.std(ddof=1)),'bootstrap_resamples':10000,'bootstrap_95pct_mean_ci':[float(np.percentile(bm,2.5)),float(np.percentile(bm,97.5))]}
def main():
 a=parse()
 if sha(a.dataset)!=SHA: raise RuntimeError('dataset SHA mismatch')
 ec,fc=load(a.emissions_checkpoint,'emissions',a.training_seed),load(a.feature_checkpoint,'feature',a.training_seed); ep,fp=ec['payload'],fc['payload']
 if ep['split_manifest']!=fp['split_manifest'] or ep['feature_columns']!=EM or fp['feature_columns']!=CTX or fp['target_columns']!=ACT: raise RuntimeError('metadata mismatch')
 df=pd.read_csv(a.dataset,low_memory=False); df['Trip']=df['Trip'].astype(str); em,fm=mdl(ec),mdl(fc); ew,fw=ep['configuration']['window_size'],fp['configuration']['window_size']
 rows=[]; Y=[]; D=[]; P=[]; series={}
 for tid in ep['split_manifest']['test_trip_ids']:
  t=df[df.Trip==str(tid)].reset_index(drop=True); fx=win(scale(t[CTX].to_numpy(float),fc),fw); act=inv(pred(fm,fx,a.batch_size),fc); s=fw
  direct=np.column_stack([t[EM[0]].to_numpy(float)[s:],t[EM[1]].to_numpy(float)[s:],t[EM[2]].to_numpy(float)[s:]])
  proxy=np.column_stack([t[EM[0]].to_numpy(float)[s:],act[:,1],act[:,0]])
  dx,px=win(scale(direct,ec),ew),win(scale(proxy,ec),ew); y=t['CO2 Emissions'].to_numpy(float)[fw+ew:]
  if len(dx)!=len(y) or len(px)!=len(y): raise RuntimeError('alignment '+str(tid))
  d=inv(pred(em,dx,a.batch_size),ec).reshape(-1); p=inv(pred(em,px,a.batch_size),ec).reshape(-1); dm,pm=met(y,d),met(y,p)
  rows.append({'trip_id':str(tid),'n_aligned_windows':len(y),'direct':dm,'proxy':pm,'proxy_minus_direct_mae':pm['mae']-dm['mae']}); Y.append(y); D.append(d); P.append(p); series[str(tid)]=(y,d,p)
 y,d,p=np.concatenate(Y),np.concatenate(D),np.concatenate(P); dg,pg=met(y,d),met(y,p); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 with (out/'tuned_ev_proxy_trip_metrics.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f); w.writerow(['trip_id','n_aligned_windows','direct_mae','direct_rmse','direct_r2','proxy_mae','proxy_rmse','proxy_r2','proxy_minus_direct_mae'])
  for r in rows: w.writerow([r['trip_id'],r['n_aligned_windows'],r['direct']['mae'],r['direct']['rmse'],r['direct']['r2'],r['proxy']['mae'],r['proxy']['rmse'],r['proxy']['r2'],r['proxy_minus_direct_mae']])
 rep=sorted((r['n_aligned_windows'],r['trip_id']) for r in rows)[(len(rows)-1)//2][1]; ry,rd,rp=series[rep]; ix=np.unique(np.linspace(0,len(ry)-1,min(len(ry),5000)).astype(int)); fig,ax=plt.subplots(figsize=(10,3.7),constrained_layout=True); ax.plot(ix,ry[ix],color='black',lw=.8,label='Observed'); ax.plot(ix,rd[ix],lw=.8,label='Measured-actuation input'); ax.plot(ix,rp[ix],lw=.8,label='Tuned predicted-actuation proxy'); ax.set(xlabel='Aligned window index within trip',ylabel='CO2 emissions (g/s)',title=f'Tuned EV composition - seed {a.training_seed}, trip {rep}'); ax.grid(alpha=.2); ax.legend(frameon=False,ncol=3); fig.savefig(out/'figure_tuned_ev_proxy_composition.png',dpi=300,bbox_inches='tight'); fig.savefig(out/'figure_tuned_ev_proxy_composition.pdf',bbox_inches='tight'); plt.close(fig)
 res={'status':'completed_tuned_ev_proxy_composition','dataset':'BMW i3','dataset_sha256':SHA,'split_seed':SPLIT_SEED,'training_seed':a.training_seed,'split_manifest':ep['split_manifest'],'test_set_used_for_model_selection':False,'checkpoints':{'emissions':{'configuration':ep['configuration'],'best_epoch':ep['checkpoint_selection']['best_epoch']},'feature':{'configuration':fp['configuration'],'best_epoch':fp['checkpoint_selection']['best_epoch']}},'alignment':{'feature_window_size':fw,'emissions_window_size':ew,'first_common_target_index':fw+ew},'global_window_weighted':{'n_windows':len(y),'direct':dg,'proxy':pg,'proxy_minus_direct_mae':pg['mae']-dg['mae']},'trip_level':{'direct_mae':summ([r['direct']['mae'] for r in rows],a.training_seed),'proxy_mae':summ([r['proxy']['mae'] for r in rows],a.training_seed+100),'proxy_minus_direct_mae':summ([r['proxy_minus_direct_mae'] for r in rows],a.training_seed+200)},'per_trip':rows,'representative_trip':rep,'interpretation_guardrail':'In-domain BMW i3 component-composition check; not evidence of causal or cross-powertrain operating-condition equivalence.'}
 (out/'tuned_ev_proxy_composition.json').write_text(json.dumps(res,indent=2),encoding='utf-8'); s1,s2,s3=res['trip_level']['direct_mae'],res['trip_level']['proxy_mae'],res['trip_level']['proxy_minus_direct_mae']; report=f"# Tuned EV feature-to-emissions composition - seed {a.training_seed}\n\nMeasured-actuation trip-mean MAE: {s1['mean']:.8f} g/s.\n\nPredicted-actuation trip-mean MAE: {s2['mean']:.8f} g/s.\n\nTrip-mean increase: {s3['mean']:.8f} g/s.\n\nWindow-weighted global MAE: {dg['mae']:.8f} -> {pg['mae']:.8f} g/s; increase {pg['mae']-dg['mae']:.8f} g/s.\n"; (out/'README.md').write_text(report,encoding='utf-8'); print(report)
if __name__=='__main__': main()
