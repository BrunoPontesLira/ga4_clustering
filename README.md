# GA4 Session Clustering

**Trace Clustering aplicado a jornadas de usuários em apps mobile — a partir de dados brutos do Google Analytics 4.**

---

## O que é este projeto?

Ferramentas de analytics tradicionais mostram *o que aconteceu* (funis, taxas de conversão, bounce rate).
Este projeto vai além: **descobre automaticamente quais perfis de jornada existem no seu app** — sem hipóteses prévias.

Cada sessão de usuário é tratada como uma sequência de eventos (um *trace*). O algoritmo agrupa sessões com jornadas similares usando K-Means sobre matrizes de transição entre eventos. O resultado é um relatório interativo que identifica padrões de comportamento, segmenta usuários e correlaciona cada perfil com conversão, engajamento e receita.

---

## Como funciona

```
JSONL (GA4 raw events)
        │
        ▼
  [1] Carregamento & normalização
        │  event_name, firebase_screen, timestamps, métricas
        ▼
  [2] Definição de atividade
        │  screen_view → nome da tela (ex: "home", "checkout")
        │  demais eventos → event_name (ex: "login", "purchase")
        ▼
  [3] Matrizes de transição por sessão
        │  Binary  — ocorreu ou não (A→B)
        │  TF      — quantas vezes ocorreu
        │  TF-IDF  — frequência ponderada pela raridade
        ▼
  [4] K-Means (k = 3 e k = 5)
        │  Testado em todas as combinações de matriz × k
        │  Silhouette Score para avaliar qualidade
        ▼
  [5] Relatório HTML interativo (Plotly)
        │  Perfil por cluster · Conversão · Engajamento
        │  Fonte de tráfego · Top transições · Silhouette plot
```

---

## Resultados gerados

O relatório HTML inclui, para cada configuração (matriz × k):

| Gráfico | O que revela |
|---|---|
| **Comparação de configurações** | Qual combinação gera os clusters mais nítidos |
| **Perfil médio por cluster** | Conversões, engajamento, receita, telas visitadas |
| **Conversão e receita por cluster** | Qual segmento converte mais e gera mais valor |
| **Engajamento por cluster** | Média e desvio padrão do tempo de sessão |
| **Fonte de tráfego por cluster** | Qual canal atrai cada perfil de usuário |
| **Top transições por cluster** | Os caminhos de navegação mais frequentes em cada grupo |
| **Silhouette plot** | Qualidade individual de cada sessão dentro do seu cluster |

---

## Estrutura do projeto

```
ga4_clustering/
├── src/
│   ├── load.py          # Leitura do JSONL e normalização de tipos
│   ├── preprocess.py    # Definição de atividade e agregação por sessão
│   ├── matrices.py      # Construção das matrizes Binary / TF / TF-IDF
│   ├── cluster.py       # K-Means + Silhouette Score
│   └── report.py        # Relatório HTML interativo com Plotly
├── results/
│   ├── matrices/        # CSVs das matrizes geradas
│   ├── clusters/        # Sessões agrupadas por cluster
│   ├── sublogs/         # Eventos originais por cluster
│   └── report.html      # Relatório final
├── run.py               # Ponto de entrada do pipeline
└── requirements.txt
```

---

## Requisitos de dados

O pipeline espera um arquivo **JSONL** (uma linha por evento) com ao menos estas colunas:

| Coluna | Descrição |
|---|---|
| `ga_session_id` | Identificador da sessão (case/trace ID) |
| `event_name` | Nome do evento GA4 |
| `event_timestamp` | Timestamp do evento |
| `event_bundle_sequence_id` | Ordem do evento dentro da sessão |
| `firebase_screen` | Nome da tela (para eventos `screen_view`) |
| `engagement_time_msec` | Tempo de engajamento do evento |
| `is_conversion` | 1 se o evento é uma conversão, 0 caso contrário |

Colunas adicionais (métricas de receita, segmento, produto, etc.) são aproveitadas automaticamente quando presentes.

---

## Como executar

```bash
# 1. Criar e ativar o ambiente virtual
python -m venv .venv
.venv/Scripts/activate        # Windows
source .venv/bin/activate     # Linux/Mac

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Apontar para o seu arquivo de dados (em run.py)
DATA_PATH = "seu_arquivo.jsonl"

# 4. Rodar o pipeline (a partir da raiz do repositório)
python -m ga4_clustering.run

# 5. Abrir o relatório
ga4_clustering/results/report.html
```

---

## Tecnologias

| | |
|---|---|
| **Python 3.10+** | Linguagem base |
| **Pandas / NumPy** | Processamento e transformação de dados |
| **Scikit-learn** | K-Means, Silhouette Score, TF-IDF |
| **Plotly** | Visualizações interativas no relatório HTML |

---

## Contexto teórico

Este projeto é uma aplicação de **Process Mining** ao domínio de analytics digital.
A abordagem de *trace clustering* — tradicional na análise de processos de negócio — é adaptada aqui para segmentação de jornadas de usuários, substituindo logs de sistemas por eventos do GA4.

A representação por **transições binárias** (A→B) captura o *caminho* percorrido pelo usuário, não apenas os eventos isolados. Isso permite identificar padrões como:
- Usuários que convertem sem explorar (jornada direta)
- Usuários que exploram muito e abandonam
- Usuários que retornam a etapas anteriores antes de converter

---

## Autor

Desenvolvido como MVP de análise de comportamento de usuários em apps mobile.
Aberto a contribuições, adaptações e discussões sobre o tema.
