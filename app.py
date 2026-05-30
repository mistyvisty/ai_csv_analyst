import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import io
import json
import requests
import warnings
warnings.filterwarnings('ignore')

# â”€â”€ Page config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="AI CSV Analyst Â· Preeti Bhardwaj",
    page_icon="ðŸŒ¿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# â”€â”€ Custom CSS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --cream: #FAF7F2;
    --forest: #2D4A3E;
    --sage:   #6B8F71;
    --moss:   #A8C5A0;
    --clay:   #C4875A;
    --ink:    #1A1A1A;
    --mist:   #E8EDE6;
}

html, body, [class*="css"], .stApp {
    background-color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
}

/* Header */
.hero { padding: 2.5rem 0 1.5rem; text-align: center; }
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: var(--forest);
    margin: 0;
    line-height: 1.1;
}
.hero h1 em { color: var(--clay); font-style: italic; }
.hero p {
    font-size: 1.05rem;
    color: var(--sage);
    margin: 0.5rem 0 0;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* Upload zone */
.upload-zone {
    border: 2px dashed var(--moss);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    background: white;
    margin: 1.5rem 0;
    transition: all 0.2s;
}

/* Cards */
.stat-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border-left: 4px solid var(--sage);
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(45,74,62,0.06);
}
.stat-card .label { font-size: 0.75rem; color: var(--sage); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
.stat-card .value { font-size: 1.6rem; font-family: 'DM Serif Display', serif; color: var(--forest); }

/* Chat messages */
.msg-user {
    background: var(--forest);
    color: white;
    border-radius: 16px 16px 4px 16px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.95rem;
    max-width: 80%;
    margin-left: auto;
}
.msg-ai {
    background: white;
    border: 1px solid var(--mist);
    border-radius: 16px 16px 16px 4px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.95rem;
    max-width: 85%;
    box-shadow: 0 2px 8px rgba(45,74,62,0.06);
    line-height: 1.6;
}
.msg-ai code {
    font-family: 'JetBrains Mono', monospace;
    background: var(--mist);
    padding: 0.1em 0.3em;
    border-radius: 4px;
    font-size: 0.85em;
    color: var(--forest);
}

/* Insight badge */
.insight-badge {
    display: inline-block;
    background: var(--mist);
    color: var(--forest);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem;
    letter-spacing: 0.03em;
}

/* Section header */
.section-label {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: var(--forest);
    border-bottom: 2px solid var(--mist);
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem;
}

/* Stremlit overrides */
.stTextInput > div > div > input {
    border: 2px solid var(--mist) !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.6rem 1rem !important;
    background: white !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--sage) !important;
    box-shadow: 0 0 0 3px rgba(107,143,113,0.15) !important;
}
.stButton > button {
    background: var(--forest) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--sage) !important;
    transform: translateY(-1px);
}
.stFileUploader { background: transparent !important; }
div[data-testid="stFileUploader"] { background: transparent !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--mist);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: var(--sage) !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: var(--forest) !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

/* Spinner */
.thinking {
    color: var(--sage);
    font-style: italic;
    font-size: 0.9rem;
    padding: 0.5rem;
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* Footer */
.footer {
    text-align: center;
    padding: 2rem;
    color: var(--sage);
    font-size: 0.8rem;
    border-top: 1px solid var(--mist);
    margin-top: 3rem;
}
.footer a { color: var(--clay); text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# â”€â”€ State init â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "df_summary" not in st.session_state:
    st.session_state.df_summary = None
if "api_key" not in st.session_state:
    import os
    st.session_state.api_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""

# â”€â”€ Helper: build data summary for LLM context â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_df_summary(df):
    summary = {}
    summary["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    summary["columns"] = list(df.columns)
    summary["dtypes"] = {col: str(df[col].dtype) for col in df.columns}
    summary["nulls"] = {col: int(df[col].isnull().sum()) for col in df.columns}
    summary["numeric_stats"] = {}
    for col in df.select_dtypes(include="number").columns:
        summary["numeric_stats"][col] = {
            "mean": round(float(df[col].mean()), 3),
            "median": round(float(df[col].median()), 3),
            "min": round(float(df[col].min()), 3),
            "max": round(float(df[col].max()), 3),
            "std": round(float(df[col].std()), 3),
        }
    summary["categorical_stats"] = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        vc = df[col].value_counts()
        summary["categorical_stats"][col] = {
            "unique": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in vc.head(5).items()}
        }
    summary["sample"] = df.head(5).to_dict(orient="records")
    return summary

# â”€â”€ Helper: call Groq API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def call_claude(messages_history, df_summary, user_question, api_key):
    system_prompt = f"""You are an expert data analyst AI assistant. The user has uploaded a CSV dataset.

DATASET SUMMARY:
{json.dumps(df_summary, indent=2, default=str)}

Your job:
1. Answer questions about this data clearly and insightfully
2. Point out interesting patterns, anomalies, or business insights
3. When asked to show a chart or visualize, respond with a JSON block describing what chart to make:
   chart_spec: {{"type": "bar or line or scatter or hist or box", "x": "column_name", "y": "column_name_or_null", "title": "Chart title", "hue": "column_name_or_null"}}
4. Be concise but insightful like a senior analyst explaining to a business stakeholder
5. Use bullet points for lists of insights
6. When you spot something interesting, lead with the insight then explain it
7. Format numbers clearly

RULES:
- Only use columns that actually exist in the dataset
- Never make up data not present in the summary
- If you cannot answer from the data, say so clearly"""

    payload = {
        "model": "llama-3.3-70b-versatile",
        "max_tokens": 500,
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages_history
        ]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# â”€â”€ Helper: render chart from JSON instruction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def render_chart(df, chart_json):
    try:
        spec = json.loads(chart_json)
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#FAF7F2')
        ax.set_facecolor('#FAF7F2')
        palette = ["#2D4A3E", "#6B8F71", "#C4875A", "#A8C5A0", "#1A1A1A"]
        ctype = spec.get("type", "bar")
        x, y, hue = spec.get("x"), spec.get("y"), spec.get("hue")
        title = spec.get("title", "")

        if ctype == "bar" and x and y:
            if x in df.columns and y in df.columns:
                data = df.groupby(x)[y].mean().reset_index().sort_values(y, ascending=False).head(15)
                ax.bar(data[x].astype(str), data[y], color=palette[0], alpha=0.85, width=0.6)
                ax.set_xlabel(x, fontsize=10, color="#6B8F71")
                ax.set_ylabel(y, fontsize=10, color="#6B8F71")
                plt.xticks(rotation=35, ha='right', fontsize=9)
        elif ctype == "hist" and x:
            if x in df.columns:
                ax.hist(df[x].dropna(), bins=25, color=palette[0], alpha=0.8, edgecolor='white')
                ax.set_xlabel(x, fontsize=10, color="#6B8F71")
                ax.set_ylabel("Count", fontsize=10, color="#6B8F71")
        elif ctype == "scatter" and x and y:
            if x in df.columns and y in df.columns:
                ax.scatter(df[x], df[y], color=palette[0], alpha=0.5, s=20)
                ax.set_xlabel(x, fontsize=10, color="#6B8F71")
                ax.set_ylabel(y, fontsize=10, color="#6B8F71")
        elif ctype == "line" and x and y:
            if x in df.columns and y in df.columns:
                d = df[[x, y]].dropna().sort_values(x)
                ax.plot(d[x], d[y], color=palette[0], linewidth=2)
                ax.set_xlabel(x, fontsize=10, color="#6B8F71")
                ax.set_ylabel(y, fontsize=10, color="#6B8F71")
        elif ctype == "box" and y:
            if y in df.columns:
                if hue and hue in df.columns:
                    groups = [g[y].dropna().values for _, g in df.groupby(hue)]
                    labels = [str(k) for k in df[hue].unique()]
                    ax.boxplot(groups, labels=labels, patch_artist=True,
                               boxprops=dict(facecolor=palette[2], alpha=0.7))
                else:
                    ax.boxplot(df[y].dropna(), patch_artist=True,
                               boxprops=dict(facecolor=palette[0], alpha=0.7))
                ax.set_ylabel(y, fontsize=10, color="#6B8F71")

        ax.set_title(title, fontsize=12, fontweight='bold', color="#2D4A3E", pad=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E8EDE6')
        ax.spines['bottom'].set_color('#E8EDE6')
        ax.tick_params(colors='#6B8F71', labelsize=9)
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

# â”€â”€ Parse AI response for chart blocks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def parse_response(text):
    import re
    chart_pattern = r"```chart\s*([\s\S]*?)```"
    charts = re.findall(chart_pattern, text)
    clean_text = re.sub(chart_pattern, "ðŸ“Š *[Chart generated below]*", text)
    return clean_text, charts

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN UI
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# Hero
st.markdown("""
<div class="hero">
    <h1>AI CSV <em>Analyst</em></h1>
    <p>Upload a CSV Â· Ask questions Â· Get instant insights & charts</p>
</div>
""", unsafe_allow_html=True)

# â”€â”€ Sidebar: API key â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.markdown("**Built by [Preeti Bhardwaj](https://mistyvisty.github.io)**")
    st.markdown("ðŸŒ¿ Powered by Groq LLaMA + Streamlit")
    if not st.session_state.api_key:
        api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        if api_key:
            st.session_state.api_key = api_key
            st.success("Key saved âœ“")

# â”€â”€ File upload â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
uploaded = st.file_uploader("Drop your CSV here", type=["csv"],
                              label_visibility="collapsed")

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.session_state.df = df
        st.session_state.df_summary = build_df_summary(df)
        if not st.session_state.messages:
            st.session_state.messages = []
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

# â”€â”€ If data loaded â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if st.session_state.df is not None:
    df = st.session_state.df

    # Quick stats bar
    num_cols = len(df.select_dtypes(include="number").columns)
    cat_cols = len(df.select_dtypes(include=["object","category"]).columns)
    nulls = df.isnull().sum().sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="label">Rows</div><div class="value">{df.shape[0]:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="label">Columns</div><div class="value">{df.shape[1]}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="label">Numeric</div><div class="value">{num_cols}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card"><div class="label">Missing values</div><div class="value">{nulls:,}</div></div>', unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["ðŸ’¬  Chat with your data", "ðŸ”  Data preview", "ðŸ“Š  Quick charts"])

    # â”€â”€ TAB 1: Chat â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab1:
        if not st.session_state.api_key:
            st.info("ðŸ‘ˆ Add your Groq API key in the sidebar to start chatting")
        else:
            # Suggested questions
            st.markdown("**Try asking:**")
            suggestions = [
                "What are the key insights from this data?",
                "Show me the distribution of " + df.select_dtypes(include="number").columns[0] if num_cols > 0 else "Show me a summary",
                "Which columns have the most missing values?",
                "What patterns do you see?",
            ]
            cols = st.columns(len(suggestions))
            for i, (col, sug) in enumerate(zip(cols, suggestions)):
                with col:
                    if st.button(sug, key=f"sug_{i}", use_container_width=True):
                        st.session_state.pending_question = sug

            st.markdown("---")

            # Chat history
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        clean_text, charts = parse_response(msg["content"])
                        st.markdown(f'<div class="msg-ai">{clean_text}</div>', unsafe_allow_html=True)
                        for chart_json in charts:
                            fig = render_chart(df, chart_json)
                            if fig:
                                st.pyplot(fig, use_container_width=True)
                                plt.close()

            # Input
            pending = st.session_state.get("pending_question", "")
            question = st.text_input("Ask anything about your data...",
                                      value=pending,
                                      placeholder="e.g. What are the top 5 categories by revenue?",
                                      key="chat_input")
            if pending:
                st.session_state.pending_question = ""

            send = st.button("Send â†’", use_container_width=False)

            if send and question.strip():
                user_msg = question.strip()
                st.session_state.messages.append({"role": "user", "content": user_msg})

                api_messages = [{"role": m["role"], "content": m["content"]}
                                 for m in st.session_state.messages]

                with st.spinner("ðŸŒ¿ Analysing..."):
                    try:
                        response = call_claude(
                            api_messages,
                            st.session_state.df_summary,
                            user_msg,
                            st.session_state.api_key
                        )
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")

                st.rerun()

            if st.session_state.messages:
                if st.button("Clear chat", use_container_width=False):
                    st.session_state.messages = []
                    st.rerun()

    # â”€â”€ TAB 2: Preview â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab2:
        st.markdown('<div class="section-label">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(50), use_container_width=True, height=380)
        st.markdown('<div class="section-label">Column Info</div>', unsafe_allow_html=True)
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(df[c].dtype) for c in df.columns],
            "Non-null": [df[c].notnull().sum() for c in df.columns],
            "Nulls": [df[c].isnull().sum() for c in df.columns],
            "Unique": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    # â”€â”€ TAB 3: Quick Charts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with tab3:
        st.markdown('<div class="section-label">Explore your data visually</div>', unsafe_allow_html=True)
        num_columns = list(df.select_dtypes(include="number").columns)
        cat_columns = list(df.select_dtypes(include=["object","category"]).columns)

        if num_columns:
            c1, c2 = st.columns(2)
            with c1:
                sel_num = st.selectbox("Numeric column", num_columns, key="qc_num")
            with c2:
                chart_type = st.selectbox("Chart type", ["Histogram", "Box plot"], key="qc_type")

            fig2, ax2 = plt.subplots(figsize=(7, 3.5))
            fig2.patch.set_facecolor('#FAF7F2')
            ax2.set_facecolor('#FAF7F2')
            if chart_type == "Histogram":
                ax2.hist(df[sel_num].dropna(), bins=25, color="#2D4A3E", alpha=0.8, edgecolor="white")
                ax2.set_title(f"Distribution of {sel_num}", fontsize=11, fontweight='bold', color="#2D4A3E")
            else:
                ax2.boxplot(df[sel_num].dropna(), patch_artist=True,
                            boxprops=dict(facecolor="#A8C5A0", alpha=0.8),
                            medianprops=dict(color="#2D4A3E", linewidth=2))
                ax2.set_title(f"Box plot â€” {sel_num}", fontsize=11, fontweight='bold', color="#2D4A3E")
            ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
            ax2.tick_params(colors='#6B8F71', labelsize=9)
            st.pyplot(fig2, use_container_width=True); plt.close()

        if cat_columns and num_columns:
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                sel_cat = st.selectbox("Category column", cat_columns, key="qc_cat")
            with c2:
                sel_val = st.selectbox("Value column", num_columns, key="qc_val")
            top_n = df.groupby(sel_cat)[sel_val].mean().sort_values(ascending=False).head(12)
            fig3, ax3 = plt.subplots(figsize=(7, 3.5))
            fig3.patch.set_facecolor('#FAF7F2'); ax3.set_facecolor('#FAF7F2')
            colors = ["#2D4A3E" if i == 0 else "#6B8F71" if i < 3 else "#A8C5A0" for i in range(len(top_n))]
            ax3.barh(top_n.index.astype(str)[::-1], top_n.values[::-1], color=colors[::-1], alpha=0.85)
            ax3.set_title(f"Avg {sel_val} by {sel_cat}", fontsize=11, fontweight='bold', color="#2D4A3E")
            ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
            ax3.tick_params(colors='#6B8F71', labelsize=9)
            st.pyplot(fig3, use_container_width=True); plt.close()

else:
    # Empty state
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:2.5rem">ðŸŒ¿</div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;color:#2D4A3E;margin:0.5rem 0">Upload a CSV to begin</div>
        <div style="color:#6B8F71;font-size:0.9rem">Sales data Â· Customer records Â· Survey results Â· Any structured CSV</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Built by <a href="https://mistyvisty.github.io">Preeti Bhardwaj</a> Â· 
    <a href="https://github.com/mistyvisty/Data-Analytics-Portfolio">GitHub</a> Â· 
    Powered by Claude AI + Streamlit
</div>
""", unsafe_allow_html=True)
