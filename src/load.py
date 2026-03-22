import json
import pandas as pd


DATA_PATH = "ga4_mobile_eventos_100_sessoes_1000_eventos.jsonl"


def load_events(path: str = DATA_PATH) -> pd.DataFrame:
    """Lê o JSONL do GA4 e retorna um DataFrame normalizado."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    # Colunas obrigatórias com tipo fixo
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df["event_bundle_sequence_id"] = pd.to_numeric(df["event_bundle_sequence_id"], errors="coerce")
    df["engagement_time_msec"] = pd.to_numeric(df["engagement_time_msec"], errors="coerce").fillna(0)
    df["is_conversion"] = pd.to_numeric(df["is_conversion"], errors="coerce").fillna(0).astype(int)

    # Colunas numéricas opcionais
    for col in ["reward_value_brl", "loan_amount_brl", "net_disbursed_brl",
                "installments", "monthly_rate_pct", "cet_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Strings vazias → None (todas as colunas string)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace("", None)

    return df
