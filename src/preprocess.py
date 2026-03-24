import pandas as pd


# Colunas de contexto da sessão (fixas por sessão — não variam por evento)
SESSION_CONTEXT_COLS = [
    "platform", "app_version", "device_os", "country", "city",
    "traffic_source", "traffic_medium", "traffic_campaign",
]


def get_activity(row: pd.Series) -> str:
    """
    Define a atividade de um evento:
    - screen_view com firebase_screen preenchido → usa o nome da tela
    - demais eventos → usa o event_name
    """
    if row["event_name"] == "screen_view" and pd.notna(row.get("firebase_screen")):
        return str(row["firebase_screen"])
    return str(row["event_name"])


def build_event_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna o event log no formato esperado pelo pipeline:
      ga_session_id | activity | event_timestamp | engagement_time_msec | is_conversion | ...contexto
    Ordenado por sessão e sequence_id.
    """
    df = df.copy()
    df["activity"] = df.apply(get_activity, axis=1)
    df = df.sort_values(["ga_session_id", "event_bundle_sequence_id"]).reset_index(drop=True)
    return df


def build_session_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega métricas por sessão para enriquecer os resultados do clustering.
    Adapta-se ao schema disponível — só inclui colunas presentes no dataset.
    Retorna um DataFrame com uma linha por sessão.
    """
    grp = df.groupby("ga_session_id")
    cols = df.columns.tolist()

    summary = pd.DataFrame({
        "user_pseudo_id":       grp["user_pseudo_id"].first(),
        "session_date":         grp["event_timestamp"].min().dt.date,
        "total_events":         grp["event_name"].count(),
        "total_engagement_sec": (grp["engagement_time_msec"].sum() / 1000).round(1),
        "conversions":          grp["is_conversion"].sum(),
        "converted":            (grp["is_conversion"].sum() > 0).astype(int),
        "unique_screens":       grp["firebase_screen"].nunique() if "firebase_screen" in cols else 0,
    })

    # Métricas de valor — usa o que estiver disponível
    if "reward_value_brl" in cols:
        summary["reward_brl"] = grp["reward_value_brl"].sum().round(2)
    if "loan_amount_brl" in cols:
        summary["loan_amount_brl"] = grp["loan_amount_brl"].max().round(2)
    if "net_disbursed_brl" in cols:
        summary["net_disbursed_brl"] = grp["net_disbursed_brl"].max().round(2)

    # Flags de eventos específicos (presença ou não na sessão)
    event_flags = {
        "has_search":        "search",
        "has_ai_suggestion": "ai_review_suggestion_view",
        "has_biometric":     "biometric_check",
        "has_contract":      "contract_sign",
        "has_error":         "app_error",
    }
    events_in_data = df["event_name"].unique()
    for flag, event in event_flags.items():
        if event in events_in_data:
            summary[flag] = grp["event_name"].apply(lambda x, e=event: int(e in x.values))

    # Contexto da sessão (primeiro valor, pois é fixo por sessão)
    for col in SESSION_CONTEXT_COLS + ["state"]:
        if col in cols:
            summary[col] = grp[col].first()

    return summary.reset_index()


def remove_outlier_sessions(
    event_log: pd.DataFrame,
    min_events: int = 3,
    max_events_percentile: float = 0.99,
) -> pd.DataFrame:
    """
    Remove sessões anômalas por volume de eventos:
    - Sessões com menos de min_events eventos (sinal insuficiente para clustering)
    - Sessões acima do percentil max_events_percentile (prováveis bots/crawlers)
    """
    sizes = event_log.groupby("ga_session_id").size()
    p_max = sizes.quantile(max_events_percentile)
    valid = sizes[(sizes >= min_events) & (sizes <= p_max)].index
    removed = len(sizes) - len(valid)
    print(f"  Sessões removidas (outliers): {removed} ({removed/len(sizes)*100:.1f}%)")
    return event_log[event_log["ga_session_id"].isin(valid)].copy()
