"""
GA4 Session Clustering — Pipeline principal
Uso: python -m ga4_clustering.run
     (executar a partir da raiz do projeto trace_clustering)
"""
import sys
import os

# Garante que a raiz do projeto está no path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ga4_clustering.src.load import load_events
from ga4_clustering.src.preprocess import build_event_log, build_session_summary
from ga4_clustering.src.matrices import build_all_matrices
from ga4_clustering.src.cluster import cluster_all
from ga4_clustering.src.report import generate_report

DATA_PATH    = "10k_eventos.jsonl"
RESULTS_PATH = "ga4_clustering/results"


def main():
    print("=" * 55)
    print("  GA4 Session Clustering")
    print("=" * 55)

    # 1. Carrega dados
    print("\n[1/5] Carregando eventos...")
    df_raw = load_events(DATA_PATH)
    print(f"  {len(df_raw)} eventos | {df_raw['ga_session_id'].nunique()} sessões")

    # 2. Pré-processa
    print("\n[2/5] Pré-processando...")
    event_log       = build_event_log(df_raw)
    session_summary = build_session_summary(df_raw)
    print(f"  Atividades únicas: {event_log['activity'].nunique()}")
    print(f"  Taxa de conversão geral: "
          f"{session_summary['converted'].mean()*100:.1f}%")

    # 3. Matrizes
    print("\n[3/5] Construindo matrizes de transição...")
    matrices = build_all_matrices(event_log, f"{RESULTS_PATH}/matrices")

    # 4. Clustering
    print("\n[4/5] Executando K-Means...")
    all_results = cluster_all(
        matrices, session_summary, event_log,
        results_path=RESULTS_PATH,
    )

    # 5. Relatório
    print("\n[5/5] Gerando relatório HTML...")
    generate_report(
        all_results,
        event_log,
        output_path=f"{RESULTS_PATH}/report.html",
    )

    print("\n" + "=" * 55)
    print(f"  Concluído! Abra: {RESULTS_PATH}/report.html")
    print("=" * 55)


if __name__ == "__main__":
    main()
