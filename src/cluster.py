import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


KS = list(range(2, 9))  # 2 a 8 — melhor combinação selecionada pelo Silhouette


def run_kmeans(matrix: pd.DataFrame, k: int, random_state: int = 42) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Executa K-Means e retorna (labels, silhouette_avg, silhouette_samples).
    """
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(matrix.values)
    sil_avg = silhouette_score(matrix.values, labels)
    sil_samples = silhouette_samples(matrix.values, labels)
    return labels, sil_avg, sil_samples


def cluster_all(
    matrices: dict,
    session_summary: pd.DataFrame,
    event_log: pd.DataFrame,
    ks: list = KS,
    results_path: str = "ga4_clustering/results",
) -> dict:
    """
    Executa clustering para todas as combinações de matriz × k.
    Salva sublogs e retorna dict com resultados.

    Retorna:
        {
          "binary_k3": {"labels": [...], "sil_avg": 0.xx, "sil_samples": [...], "summary": df},
          ...
        }
    """
    all_results = {}

    for matrix_name, matrix in matrices.items():
        for k in ks:
            key = f"{matrix_name}_k{k}"
            print(f"  Clustering {key}...")

            labels, sil_avg, sil_samples = run_kmeans(matrix, k)

            # Associa cluster ao session_summary
            cluster_map = pd.Series(labels, index=matrix.index, name="cluster")
            summary_clustered = session_summary.set_index("ga_session_id").join(cluster_map)
            summary_clustered = summary_clustered.reset_index()

            # Salva CSV por cluster + sublog de eventos
            for cluster_id in range(k):
                cluster_sessions = summary_clustered[summary_clustered["cluster"] == cluster_id]
                cluster_sessions.to_csv(
                    f"{results_path}/clusters/{key}_cluster{cluster_id}.csv", index=False
                )

                sublog = event_log[event_log["ga_session_id"].isin(cluster_sessions["ga_session_id"])]
                sublog.to_csv(
                    f"{results_path}/sublogs/{key}_cluster{cluster_id}_events.csv", index=False
                )

            print(f"    Silhouette avg: {sil_avg:.3f} | clusters: {k}")

            all_results[key] = {
                "matrix_name": matrix_name,
                "k": k,
                "labels": labels,
                "sil_avg": sil_avg,
                "sil_samples": sil_samples,
                "summary": summary_clustered,
                "matrix": matrix,
            }

    return all_results
