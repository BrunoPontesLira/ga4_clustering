import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Paleta de cores por cluster
CLUSTER_COLORS = px.colors.qualitative.Set2


def _silhouette_chart(sil_samples: np.ndarray, labels: np.ndarray, k: int, title: str) -> go.Figure:
    """Silhouette plot clássico: barras horizontais empilhadas por cluster."""
    fig = go.Figure()

    sil_avg = float(np.mean(sil_samples))
    GAP = max(10, int(len(sil_samples) * 0.015))  # espaço entre clusters

    y_offset = 0
    cluster_centers = []   # posição Y central de cada cluster (para rótulo)
    tick_vals = []
    tick_texts = []

    for i in range(k):
        vals = np.sort(sil_samples[labels == i])
        n = len(vals)
        mean_i = float(np.mean(vals))
        pct_neg = float((vals < 0).mean() * 100)
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

        y_positions = list(range(y_offset, y_offset + n))
        center_y = y_offset + n / 2
        cluster_centers.append(center_y)

        fig.add_trace(go.Bar(
            x=vals.tolist(),
            y=y_positions,
            orientation="h",
            base=0,
            width=1,
            marker=dict(color=color, line=dict(width=0)),
            name=f"Cluster {i}",
            showlegend=False,
            hovertemplate=(
                f"<b>Cluster {i}</b><br>"
                f"n = {n} sessões<br>"
                f"Média: {mean_i:.3f}<br>"
                f"Negativos: {pct_neg:.1f}%<br>"
                "Score: %{x:.3f}<extra></extra>"
            ),
        ))

        # Rótulo à esquerda do cluster
        label = f"<b>C{i}</b>  n={n}  μ={mean_i:.3f}"
        if pct_neg > 0:
            label += f"  ⚠{pct_neg:.0f}%"
        tick_vals.append(center_y)
        tick_texts.append(label)

        y_offset += n + GAP

    # Linha vertical: média global
    fig.add_vline(
        x=sil_avg,
        line_dash="dash",
        line_color="crimson",
        line_width=2,
        annotation_text=f"Média: {sil_avg:.3f}",
        annotation_position="top right",
        annotation_font=dict(color="crimson", size=11),
    )

    # Linha vertical: zero
    fig.add_vline(
        x=0,
        line_color="#bbb",
        line_width=1,
    )

    x_min = min(-0.15, float(sil_samples.min()) - 0.05)
    x_max = min(1.0,   float(sil_samples.max()) + 0.08)

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>"
                 "Cada faixa = um cluster · Largura = score de cada sessão · "
                 "⚠ = % com score negativo (mal agrupado)</sup>",
            font=dict(size=14),
        ),
        xaxis=dict(
            title="Silhouette Coefficient",
            range=[x_min, x_max],
            gridcolor="#eee",
            zeroline=False,
            tickformat=".1f",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_texts,
            showgrid=False,
            zeroline=False,
        ),
        template="plotly_white",
        height=max(350, k * 110),
        showlegend=False,
        margin=dict(l=160, r=80, t=90, b=60),
    )
    return fig


def _conversion_by_cluster(summary: pd.DataFrame, k: int, title: str) -> go.Figure:
    """Taxa de conversão e valor médio (receita/empréstimo) por cluster."""
    # Detecta coluna de valor disponível
    value_col, value_label = None, None
    for col, label in [("reward_brl", "Recompensa média (R$)"),
                       ("net_disbursed_brl", "Valor liberado médio (R$)"),
                       ("loan_amount_brl", "Valor solicitado médio (R$)")]:
        if col in summary.columns:
            value_col, value_label = col, label
            break

    agg = {"sessoes": ("ga_session_id", "count"), "conv_rate": ("converted", "mean")}
    if value_col:
        agg["valor_medio"] = (value_col, "mean")

    grp = summary.groupby("cluster").agg(**agg).reset_index()
    grp["conv_pct"] = (grp["conv_rate"] * 100).round(1)
    grp["cluster_label"] = grp["cluster"].apply(lambda x: f"Cluster {x}")

    use_secondary = value_col is not None
    fig = make_subplots(specs=[[{"secondary_y": use_secondary}]])

    fig.add_trace(go.Bar(
        x=grp["cluster_label"], y=grp["conv_pct"],
        name="Taxa de conversão (%)",
        marker_color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in grp["cluster"]],
        text=grp["conv_pct"].apply(lambda v: f"{v}%"),
        textposition="outside",
    ), secondary_y=False)

    if use_secondary:
        fig.add_trace(go.Scatter(
            x=grp["cluster_label"], y=grp["valor_medio"].round(2),
            name=value_label,
            mode="lines+markers",
            marker=dict(size=10, color="crimson"),
            line=dict(color="crimson", dash="dot"),
        ), secondary_y=True)
        fig.update_yaxes(title_text=value_label, secondary_y=True)

    fig.update_layout(
        title=title, template="plotly_white", height=380,
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_yaxes(title_text="Taxa de conversão (%)", secondary_y=False)
    return fig


def _top_transitions(event_log: pd.DataFrame, summary: pd.DataFrame, k: int, title: str) -> go.Figure:
    """Top 10 transições por cluster (barras horizontais empilhadas)."""
    merged = event_log.merge(summary[["ga_session_id", "cluster"]], on="ga_session_id")
    merged = merged.sort_values(["ga_session_id", "event_bundle_sequence_id"])

    rows = []
    for session_id, grp in merged.groupby("ga_session_id", sort=False):
        acts = grp["activity"].tolist()
        cluster = grp["cluster"].iloc[0]
        for i in range(len(acts) - 1):
            rows.append({"transition": f"{acts[i]}→{acts[i+1]}", "cluster": cluster})
    trans = pd.DataFrame(rows)

    rows = []
    for c in range(k):
        top = (trans[trans["cluster"] == c]["transition"]
               .value_counts().head(10).reset_index())
        top.columns = ["transition", "count"]
        top["cluster"] = c
        rows.append(top)

    df_top = pd.concat(rows, ignore_index=True)

    fig = px.bar(
        df_top, x="count", y="transition", color="cluster",
        facet_col="cluster", orientation="h",
        color_discrete_sequence=CLUSTER_COLORS,
        title=title, template="plotly_white",
        height=max(400, len(df_top["transition"].unique()) * 22),
        labels={"count": "Ocorrências", "transition": ""},
    )
    fig.update_layout(showlegend=False)
    fig.for_each_annotation(lambda a: a.update(text=f"Cluster {a.text.split('=')[1]}"))
    return fig


def _source_distribution(summary: pd.DataFrame, k: int, title: str) -> go.Figure:
    """Distribuição de traffic_source por cluster."""
    grp = summary.groupby(["cluster", "traffic_source"]).size().reset_index(name="count")
    grp["cluster_label"] = grp["cluster"].apply(lambda x: f"Cluster {x}")
    fig = px.bar(
        grp, x="cluster_label", y="count", color="traffic_source",
        title=title, template="plotly_white", height=380,
        labels={"count": "Sessões", "cluster_label": "", "traffic_source": "Fonte"},
        text_auto=True,
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    return fig


def _engagement_boxplot(summary: pd.DataFrame, k: int, title: str) -> go.Figure:
    """Barras de média de engajamento por cluster com desvio padrão e contagem de sessões."""
    grp = summary.groupby("cluster")["total_engagement_sec"].agg(
        media="mean", desvio="std", sessoes="count"
    ).reset_index().fillna(0)
    grp["media"] = grp["media"].round(1)
    grp["desvio"] = grp["desvio"].round(1)
    grp["label"] = grp["cluster"].apply(lambda x: f"Cluster {x}")
    grp["texto"] = grp.apply(
        lambda r: f"{r['media']:.0f}s ± {r['desvio']:.0f}s<br>{int(r['sessoes'])} sessões", axis=1
    )

    fig = go.Figure()
    for _, row in grp.iterrows():
        c = int(row["cluster"])
        fig.add_trace(go.Bar(
            x=[row["label"]],
            y=[row["media"]],
            error_y=dict(type="data", array=[row["desvio"]], visible=True, color="#888"),
            name=row["label"],
            marker_color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
            text=row["texto"],
            textposition="outside",
            textfont=dict(size=12),
            hovertemplate=(
                f"<b>{row['label']}</b><br>"
                f"Média: {row['media']:.1f}s<br>"
                f"Desvio: ±{row['desvio']:.1f}s<br>"
                f"Sessões: {int(row['sessoes'])}<extra></extra>"
            ),
        ))

    y_max = (grp["media"] + grp["desvio"]).max()
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>Barra = média | traço = desvio padrão | rótulo = nº de sessões</sup>",
            font=dict(size=14),
        ),
        yaxis=dict(
            title="Engajamento médio (s)",
            range=[0, y_max * 1.35],
            gridcolor="#eee",
        ),
        xaxis=dict(title=""),
        template="plotly_white",
        height=400,
        showlegend=False,
        bargap=0.35,
    )
    return fig


def _silhouette_comparison(all_results: dict) -> go.Figure:
    """Gráfico de barras agrupadas: representação × k, com valor anotado."""
    rows = [
        {"matrix": v["matrix_name"], "k": f"k = {v['k']}", "silhouette": round(v["sil_avg"], 4)}
        for v in all_results.values()
    ]
    df = pd.DataFrame(rows)

    ks = sorted(df["k"].unique())
    matrices = sorted(df["matrix"].unique())
    best_val = df["silhouette"].max()

    fig = go.Figure()
    for i, k_val in enumerate(ks):
        subset = df[df["k"] == k_val].set_index("matrix").reindex(matrices)
        colors = [
            "#2ca02c" if v == best_val else CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            for v in subset["silhouette"]
        ]
        fig.add_trace(go.Bar(
            name=k_val,
            x=matrices,
            y=subset["silhouette"].tolist(),
            text=[f"{v:.3f}" for v in subset["silhouette"]],
            textposition="outside",
            textfont=dict(size=13),
            marker_color=colors,
            marker_line_width=0,
        ))

    # Linha de referência: média geral
    mean_all = df["silhouette"].mean()
    fig.add_hline(
        y=mean_all, line_dash="dash", line_color="#888", line_width=1,
        annotation_text=f"  Média: {mean_all:.3f}",
        annotation_position="right",
        annotation_font=dict(color="#888", size=10),
    )

    y_max = df["silhouette"].max()
    fig.update_layout(
        title=dict(
            text="Comparação de configurações — Silhouette Score<br>"
                 "<sup>Quanto maior, mais bem separados estão os clusters. "
                 "Verde = melhor configuração.</sup>",
            font=dict(size=14),
        ),
        xaxis=dict(title="Representação"),
        yaxis=dict(
            title="Silhouette Score",
            range=[0, y_max * 1.25],
            gridcolor="#eee",
        ),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.1,
        template="plotly_white",
        height=360,
        legend=dict(title="k", orientation="h", y=-0.18),
        margin=dict(l=60, r=80, t=90, b=60),
    )
    return fig


def _cluster_profile_table(summary: pd.DataFrame, k: int) -> go.Figure:
    """Tabela de perfil médio por cluster — adapta-se às colunas disponíveis."""
    all_cols = {
        "conversions":        "Conversões",
        "total_events":       "Total eventos",
        "total_engagement_sec": "Engajamento (s)",
        "unique_screens":     "Telas únicas",
        "reward_brl":         "Recompensa (R$)",
        "loan_amount_brl":    "Valor empréstimo (R$)",
        "net_disbursed_brl":  "Valor liberado (R$)",
        "has_ai_suggestion":  "Usou IA (%)",
        "has_biometric":      "Biometria (%)",
        "has_contract":       "Contrato (%)",
        "has_error":          "Teve erro (%)",
    }
    pct_cols = {"has_ai_suggestion", "has_biometric", "has_contract", "has_error"}

    cols = [c for c in all_cols if c in summary.columns]
    profile = summary.groupby("cluster")[cols].mean().round(2).reset_index()
    for c in pct_cols:
        if c in profile.columns:
            profile[c] = (profile[c] * 100).round(1)
    profile.columns = ["Cluster"] + [all_cols[c] for c in cols]
    profile["Cluster"] = profile["Cluster"].apply(lambda x: f"Cluster {x}")

    fig = go.Figure(go.Table(
        header=dict(
            values=list(profile.columns),
            fill_color="#4C72B0",
            font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=[profile[c].tolist() for c in profile.columns],
            fill_color=[["#f0f4ff", "#e6f0ff"] * (len(profile) // 2 + 1)],
            align="center",
            font=dict(size=12),
        ),
    ))
    fig.update_layout(title="Perfil médio por cluster", height=250 + k * 30)
    return fig


def _fig_to_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False)


def generate_report(
    all_results: dict,
    event_log: pd.DataFrame,
    output_path: str = "ga4_clustering/results/report.html",
    best_key: str = None,
) -> None:
    """
    Gera um relatório HTML completo com todos os gráficos.
    best_key: chave do resultado a destacar como "melhor configuração".
              Se None, usa o de maior silhouette.
    """
    if best_key is None:
        best_key = max(all_results, key=lambda k: all_results[k]["sil_avg"])

    best = all_results[best_key]
    k = best["k"]
    summary = best["summary"]

    sections = []

    # ── Cabeçalho
    sections.append(f"""
    <div class="section">
      <h2>Melhor configuração: <code>{best_key}</code></h2>
      <p>Silhouette médio: <strong>{best['sil_avg']:.4f}</strong> &nbsp;|&nbsp;
         k = <strong>{k}</strong> clusters &nbsp;|&nbsp;
         {len(summary)} sessões analisadas</p>
    </div>
    """)

    # ── Comparação geral de silhouette
    sections.append('<div class="section"><h2>Comparação de configurações</h2>')
    sections.append(_fig_to_html(_silhouette_comparison(all_results)))
    sections.append("</div>")

    # ── Perfil dos clusters (tabela)
    sections.append('<div class="section"><h2>Perfil dos clusters</h2>')
    sections.append(_fig_to_html(_cluster_profile_table(summary, k)))
    sections.append("</div>")

    # ── Conversão e recompensa
    sections.append('<div class="section"><h2>Conversão e recompensa por cluster</h2>')
    sections.append(_fig_to_html(_conversion_by_cluster(summary, k, "Taxa de conversão e recompensa média")))
    sections.append("</div>")

    # ── Engajamento
    sections.append('<div class="section"><h2>Engajamento por cluster</h2>')
    sections.append(_fig_to_html(_engagement_boxplot(summary, k, "Tempo de engajamento por cluster")))
    sections.append("</div>")

    # ── Fonte de tráfego
    sections.append('<div class="section"><h2>Fonte de tráfego por cluster</h2>')
    sections.append(_fig_to_html(_source_distribution(summary, k, "Distribuição de traffic_source")))
    sections.append("</div>")

    # ── Top transições
    sections.append('<div class="section"><h2>Top transições por cluster</h2>')
    sections.append(_fig_to_html(_top_transitions(event_log, summary, k, "Top 10 transições mais frequentes")))
    sections.append("</div>")

    # ── Silhouette
    sections.append('<div class="section"><h2>Silhouette por cluster</h2>')
    sections.append(_fig_to_html(_silhouette_chart(
        best["sil_samples"], best["labels"], k,
        f"Silhouette — {best_key}"
    )))
    sections.append("</div>")

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>GA4 Session Clustering — Relatório</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; background: #f8f9fa; margin: 0; padding: 0; }}
    header {{ background: #1a237e; color: white; padding: 24px 40px; }}
    header h1 {{ margin: 0; font-size: 1.6rem; }}
    header p  {{ margin: 4px 0 0; opacity: .8; font-size: .95rem; }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 24px 20px; }}
    .section {{ background: white; border-radius: 10px; padding: 24px;
                margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .section h2 {{ margin-top: 0; color: #1a237e; font-size: 1.1rem; }}
    code {{ background: #eef; padding: 2px 6px; border-radius: 4px; font-size: .9em; }}
  </style>
</head>
<body>
  <header>
    <h1>GA4 Session Clustering</h1>
    <p>Análise de jornadas de usuários — app mobile de reviews</p>
  </header>
  <div class="container">
    {''.join(sections)}
  </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Relatório gerado: {output_path}")
