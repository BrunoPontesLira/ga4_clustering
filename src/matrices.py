import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize


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


def _filter_rare_transitions(matrix: pd.DataFrame, min_event_freq: int) -> pd.DataFrame:
    """Remove colunas (transições) que aparecem em menos de min_event_freq sessões."""
    col_support = (matrix > 0).sum(axis=0)
    return matrix.loc[:, col_support >= min_event_freq]


def _normalize_l2(matrix: pd.DataFrame) -> pd.DataFrame:
    """Normalização L2 por linha — garante escala uniforme entre sessões curtas e longas."""
    normed = normalize(matrix.values, norm="l2")
    return pd.DataFrame(normed, index=matrix.index, columns=matrix.columns)


def build_all_matrices(
    event_log: pd.DataFrame,
    results_path: str = "ga4_clustering/results/matrices",
    min_event_freq: int = 5,
) -> dict:
    """
    Constrói e salva as três matrizes. Retorna dict com os DataFrames prontos para clustering.

    Aplica:
    - Filtro de transições raras (min_event_freq) em todas as matrizes
    - Normalização L2 nas matrizes Binary e TF (TF-IDF já é normalizado internamente)
    """
    binary = build_binary_matrix(event_log)
    tf     = build_tf_matrix(event_log)
    tfidf  = build_tfidf_matrix(tf)

    # Filtra transições raras
    binary = _filter_rare_transitions(binary, min_event_freq)
    tf     = _filter_rare_transitions(tf,     min_event_freq)
    tfidf  = _filter_rare_transitions(tfidf,  min_event_freq)

    # Normalização L2 para Binary e TF (equaliza sessões curtas vs. longas)
    binary_norm = _normalize_l2(binary)
    tf_norm     = _normalize_l2(tf)

    binary.to_csv(f"{results_path}/binary.csv")
    tf.to_csv(f"{results_path}/tf.csv")
    tfidf.to_csv(f"{results_path}/tfidf.csv")

    n_trans_before = build_tf_matrix(event_log).shape[1]
    print(f"  Matrizes geradas: {binary.shape[0]} sessões × {binary.shape[1]} transições "
          f"(filtradas {n_trans_before - binary.shape[1]} raras)")

    return {"binary": binary_norm, "tf": tf_norm, "tfidf": tfidf}
