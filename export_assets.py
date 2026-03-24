"""
Gera imagens estáticas (PNG) dos principais gráficos para o README do GitHub.
Uso: python export_assets.py --data "data for testing/10k_eventos.jsonl"
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ga4_clustering.src.load import load_events
from ga4_clustering.src.preprocess import build_event_log, build_session_summary, remove_outlier_sessions
from ga4_clustering.src.matrices import build_all_matrices
from ga4_clustering.src.cluster import cluster_all
from ga4_clustering.src.report import (
    _silhouette_comparison,
    _cluster_profile_table,
    _conversion_by_cluster,
    _engagement_boxplot,
    _silhouette_chart,
    _top_transitions,
)

ASSETS_DIR = "assets"


def export(fig, name: str, width: int = 1000, height: int = None):
    path = os.path.join(ASSETS_DIR, f"{name}.png")
    kwargs = {"width": width}
    if height:
        kwargs["height"] = height
    fig.write_image(path, scale=2, **kwargs)
    print(f"  Salvo: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data for testing/10k_eventos.jsonl")
    parser.add_argument("--min-events", type=int, default=3)
    parser.add_argument("--min-event-freq", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(ASSETS_DIR, exist_ok=True)

    print("Carregando dados...")
    df_raw = load_events(args.data)
    event_log = build_event_log(df_raw)
    event_log = remove_outlier_sessions(event_log, min_events=args.min_events)
    session_summary = build_session_summary(
        df_raw[df_raw["ga_session_id"].isin(event_log["ga_session_id"])]
    )

    print("Construindo matrizes...")
    matrices = build_all_matrices(
        event_log,
        results_path="results/matrices",
        min_event_freq=args.min_event_freq,
    )

    print("Executando clustering...")
    all_results = cluster_all(
        matrices, session_summary, event_log,
        results_path="results",
    )

    best_key = max(all_results, key=lambda k: all_results[k]["sil_avg"])
    best = all_results[best_key]
    k = best["k"]
    summary = best["summary"]

    print(f"\nMelhor configuração: {best_key} (Silhouette: {best['sil_avg']:.3f})")
    print("\nExportando imagens para assets/...")

    export(_silhouette_comparison(all_results), "silhouette_comparison", height=380)
    export(_cluster_profile_table(summary, k), "cluster_profile", height=250 + k * 30)
    export(_conversion_by_cluster(summary, k, "Conversão por cluster"), "conversion_by_cluster", height=400)
    export(_engagement_boxplot(summary, k, "Engajamento por cluster"), "engagement_by_cluster", height=420)
    export(_silhouette_chart(best["sil_samples"], best["labels"], k, f"Silhouette — {best_key}"),
           "silhouette_best", height=max(400, k * 120))
    export(_top_transitions(event_log, summary, k, "Top transições por cluster"),
           "top_transitions", height=500)

    print(f"\nConcluído — {len(os.listdir(ASSETS_DIR))} imagens em assets/")


if __name__ == "__main__":
    main()
