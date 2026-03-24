"""
GA4 Session Clustering — Pipeline principal

Uso:
  python -m ga4_clustering.run --data 10k_eventos.jsonl
  python -m ga4_clustering.run --data 10k_eventos.jsonl --k-values 2 3 4 5 6
  python -m ga4_clustering.run --data 10k_eventos.jsonl --results ga4_clustering/results
"""
import sys
import os
import argparse

# Garante que a raiz do projeto está no path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ga4_clustering.src.load import load_events
from ga4_clustering.src.preprocess import build_event_log, build_session_summary, remove_outlier_sessions
from ga4_clustering.src.matrices import build_all_matrices
from ga4_clustering.src.cluster import cluster_all
from ga4_clustering.src.report import generate_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GA4 Session Clustering — agrupa sessões por padrão de navegação."
    )
    parser.add_argument(
        "--data",
        default=os.environ.get("GA4_DATA_PATH", "10k_eventos.jsonl"),
        help="Caminho para o arquivo JSONL de eventos GA4 (default: GA4_DATA_PATH env ou 10k_eventos.jsonl)",
    )
    parser.add_argument(
        "--results",
        default=os.environ.get("GA4_RESULTS_PATH", "ga4_clustering/results"),
        help="Diretório de saída dos resultados (default: ga4_clustering/results)",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=list(range(2, 9)),
        metavar="K",
        help="Valores de k para o K-Means (default: 2 3 4 5 6 7 8)",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=3,
        help="Mínimo de eventos por sessão (sessões menores são removidas, default: 3)",
    )
    parser.add_argument(
        "--min-event-freq",
        type=int,
        default=5,
        help="Frequência mínima de um evento para entrar na matriz (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  GA4 Session Clustering")
    print("=" * 55)
    print(f"  Dados:    {args.data}")
    print(f"  k values: {args.k_values}")

    # 1. Carrega dados
    print("\n[1/5] Carregando eventos...")
    df_raw = load_events(args.data)
    print(f"  {len(df_raw)} eventos | {df_raw['ga_session_id'].nunique()} sessões")

    # 2. Pré-processa
    print("\n[2/5] Pré-processando...")
    event_log = build_event_log(df_raw)
    event_log = remove_outlier_sessions(event_log, min_events=args.min_events)
    session_summary = build_session_summary(df_raw[df_raw["ga_session_id"].isin(event_log["ga_session_id"])])
    print(f"  Atividades únicas: {event_log['activity'].nunique()}")
    print(f"  Taxa de conversão geral: "
          f"{session_summary['converted'].mean()*100:.1f}%")

    # 3. Matrizes
    print("\n[3/5] Construindo matrizes de transição...")
    matrices = build_all_matrices(
        event_log,
        results_path=f"{args.results}/matrices",
        min_event_freq=args.min_event_freq,
    )

    # 4. Clustering
    print("\n[4/5] Executando K-Means...")
    all_results = cluster_all(
        matrices, session_summary, event_log,
        ks=args.k_values,
        results_path=args.results,
    )

    # 5. Relatório
    print("\n[5/5] Gerando relatório HTML...")
    generate_report(
        all_results,
        event_log,
        output_path=f"{args.results}/report.html",
    )

    print("\n" + "=" * 55)
    print(f"  Concluído! Abra: {args.results}/report.html")
    print("=" * 55)


if __name__ == "__main__":
    main()
