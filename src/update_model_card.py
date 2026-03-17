import json
from datetime import date
from pathlib import Path

MODEL_CARD_JSON = Path("model_card.json")
MODEL_CARD_MD = Path("MODEL_CARD.md")
METRICS_JSON = Path("metrics.json")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def gather_results(base_path: Path) -> list:
    print("list jsons")
    
    results = []
    for p in base_path:
        results.append((p,load_json(p)))
    return results

def build_markdown(card: dict) -> str:
    md = []
    md.append("# Model Card — Credible Co2\n")
    md.append("Decarbonizing road transport requires consistent and transparent methods for comparing CO2 emissions across vehicle technologies. This work proposes a machine learning–based framework for like-for-like operational assessment of internal combustion engine vehicles (ICEVs) and electric vehicles (EVs) under identical, real-world driving conditions.")
    md.append("## 1. General Information:")
    md.append("Here we provide a list of the LSTM models and datasets they were trained to predict Carbon Dioxide (Co2) emissions (or features related to it).\n")
    md.append("|    Dataset |   Entries |   Type    |")
    md.append("|    :------:    |   :------:    |   :------:    | ")
    md.append("|    Infiniti QX50   |   `377149`    |   ICEV    |")    
    md.append("|    Chevrolet Blazer   |   `108678`    |   ICEV    |")    
    md.append("|    Chrysler Pacifica   |   `183996`    |   ICEV    |")        
    md.append("|    BMW i3 (ieee)   |   `1094794`    |   EV    |")        
    md.append("\n")
    
    md.append("LSTMs:")
    md.append("- Electrical Vehicle (EV): Between 1-4 layers with 32 hidden units per layer + layer norm and residual connections.")
    md.append("- Internal Combustion Engine Vehicle (ICEV): Between 1-4 layers with 64 hidden units per layer + layer norm and residual connections.")

    md.append("## 2. Intended uses")
    md.append("The overall methodology (models, training scritps, data processing routines, etc) is intended to researchers or enthusiasts who may feel inspired to build upon this project to carry out work involving Co2 prediction with vehicle (time-series) data or related.\n")
    
    md.append("## 3. Evaluation scenarios")
    md.append("- 3.1 - Domain Specific Training:")
    md.append(" - Emission model:")
    md.append(" - Feature model:")
    md.append("- 3.2 - Proxy Validation:")    
    md.append(" - lorem ipsum:")
    md.append("- 3.3 - Test-Time Conterfactual Analysis:")    
    md.append(" - lorem ipsum:")
    
    
    md.append("## 4. Results")
        
    # md.append(f"*Nome do modelo:* {card['model_details']['name']}  ")
    # md.append(f"*Versão:* {card['model_details']['version']}  ")
    # md.append(f"*Status:* {card['model_details']['status']}  ")
    # md.append(f"*Tipo:* {card['model_details']['type']}  \n")
    # md.append(f"{card['model_details']['description']}\n")

    # for item in card["intended_use"]["use_cases"]:
    #     md.append(f"- {item}")
    # md.append("\n### Não usar para")
    # for item in card["intended_use"]["out_of_scope"]:
    #     md.append(f"- {item}")
    # md.append("")

    # md.append("## 3. Dados")
    # md.append("### Fonte dos dados")
    # for src in card["data"]["sources"]:
    #     md.append(f"- {src}")
    # md.append("")
    # md.append("### Janela dos dados")
    # md.append(
    #     f"- Treino: {card['data']['train_period'][0]} a {card['data']['train_period'][1]}"
    # )
    # md.append(
    #     f"- Validação: {card['data']['validation_period'][0]} a {card['data']['validation_period'][1]}"
    # )
    # md.append(
    #     f"- Teste: {card['data']['test_period'][0]} a {card['data']['test_period'][1]}"
    # )
    # md.append("")
    # md.append("### Features principais")
    # for feat in card["data"]["features"]:
    #     md.append(f"- {feat}")
    # md.append("")
    # md.append("### Target")
    # md.append(f"- {card['data']['target']}\n")

    # overall = card["metrics"]["overall"]
    # md.append("## 4. Métricas")
    # md.append("### Geral")
    # md.append(f"- AUC: {overall['auc']:.2f}")
    # md.append(f"- Accuracy: {overall['accuracy']:.2f}")
    # md.append(f"- Precision: {overall['precision']:.2f}")
    # md.append(f"- Recall: {overall['recall']:.2f}")
    # md.append(f"- F1: {overall['f1']:.2f}\n")

    # md.append("### Por segmento")
    # md.append("| Segmento | AUC | Recall |")
    # md.append("|---|---:|---:|")
    # for row in card["metrics"]["by_segment"]:
    #     md.append(f"| {row['segment']} | {row['auc']:.2f} | {row['recall']:.2f} |")
    # md.append("")

    # md.append("## 5. Limitações")
    # for item in card["limitations"]:
    #     md.append(f"- {item}")
    # md.append("")

    # md.append("## 6. Riscos e considerações éticas")
    # for item in card["ethical_considerations"]:
    #     md.append(f"- {item}")
    # md.append("")

    # md.append("## 7. Monitoramento")
    # md.append(f"- Frequência de reavaliação: {card['monitoring']['review_frequency']}")
    # md.append(
    #     f"- Alertas para drift de features: {'sim' if card['monitoring']['feature_drift_alerts'] else 'não'}"
    # )
    # md.append(
    #     f"- Re-treino automático: {'sim' if card['monitoring']['auto_retraining'] else 'não'}"
    # )
    # md.append("")

    # md.append("## 8. Responsáveis")
    # md.append(f"- Time: {card['owners']['team']}")
    # md.append(f"- Contato: {card['owners']['contact']}")
    # md.append(f"- Última atualização: {card['owners']['last_updated']}")

    return "\n".join(md) + "\n"


def main() -> None:
    #card = load_json(MODEL_CARD_JSON)
    #metrics = load_json(METRICS_JSON)

    #card["metrics"]["overall"] = metrics["overall"]
    #card["metrics"]["by_segment"] = metrics["by_segment"]
    #card["owners"]["last_updated"] = str(date.today())

    #save_json(MODEL_CARD_JSON, card)
    MODEL_CARD_MD.write_text(build_markdown(None), encoding="utf-8")

    print("Model card atualizado com sucesso.")

main()