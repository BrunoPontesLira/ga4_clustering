import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer


def _build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada sessão, cria pares de transição consecutivos (A→B).
    Retorna DataFrame com colunas: ga_session_id, transition
    """
    rows = []
    for session_id, grp in df.groupby("ga_session_id", sort=False):
        acts = grp["activity"].tolist()
        for i in range(len(acts) - 1):
            rows.append({"ga_session_id": session_id, "transition": f"{acts[i]}→{acts[i+1]}"})
    return pd.DataFrame(rows)


def build_binary_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz binária: 1 se a transição ocorreu na sessão, 0 caso contrário."""
    trans = _build_transitions(df)
    matrix = pd.pivot_table(
        trans,
        index="ga_session_id",
        columns="transition",
        aggfunc=lambda x: 1,
        fill_value=0,
    )
    matrix.columns.name = None
    return matrix


def build_tf_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz TF: contagem de ocorrências de cada transição por sessão."""
    trans = _build_transitions(df)
    matrix = pd.pivot_table(
        trans,
        index="ga_session_id",
        columns="transition",
        aggfunc=len,
        fill_value=0,
    )
    matrix.columns.name = None
    return matrix


def build_tfidf_matrix(tf_matrix: pd.DataFrame) -> pd.DataFrame:
    """Matriz TF-IDF a partir da matriz TF."""
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf_values = transformer.fit_transform(tf_matrix.values)
    tfidf = pd.DataFrame(
        tfidf_values.toarray(),
        index=tf_matrix.index,
        columns=tf_matrix.columns,
    )
    return tfidf


def build_all_matrices(event_log: pd.DataFrame, results_path: str = "ga4_clustering/results/matrices") -> dict:
    """Constrói e salva as três matrizes. Retorna dict com os DataFrames."""
    binary = build_binary_matrix(event_log)
    tf     = build_tf_matrix(event_log)
    tfidf  = build_tfidf_matrix(tf)

    binary.to_csv(f"{results_path}/binary.csv")
    tf.to_csv(f"{results_path}/tf.csv")
    tfidf.to_csv(f"{results_path}/tfidf.csv")

    print(f"  Matrizes geradas: {binary.shape[0]} sessões × {binary.shape[1]} transições")

    return {"binary": binary, "tf": tf, "tfidf": tfidf}
