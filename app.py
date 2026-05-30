import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import re
import requests
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI CSV Analyst · Preeti Bhardwaj",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --cream: #FAF7F2; --forest: #2D4A3E; --sage: #6B8F71;
  --moss: #A8C5A0; --clay: #C4875A; --ink: #1A1A1A; --mist: #E8EDE6;
}

html, body, [class*="css"], .stApp {
  background-color: var(--cream) !important;
  font-family: 'DM Sans', sans-serif;
  color: var(--ink);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

.hero { padding: 2rem 0 1.2rem; text-align: center; }
.hero h1 { font-family: 'DM Serif Display', serif; font-size: 2.8rem; color: var(--forest); margin: 0; line-height: 1.1; }
.hero h1 em { color: var(--clay); font-style: italic; }
.hero p { font-size: 1rem; color: var(--sage); margin: 0.4rem 0 0; font-weight: 300; }

.stat-card { background: white; border-radius: 12px; padding: 1rem 1.4rem; border-left: 4px solid var(--sage); box-shadow: 0 2px 8px rgba(45,74,62,0.06); }
.stat-card .label { font-size: 0.72rem; color: var(--sage); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
.stat-card .value { font-size: 1.5rem; font-family: 'DM Serif Display', serif; color: var(--forest); }

.msg-user { background: var(--forest); color: white; border-radius: 16px 16px 4px 16px; padding: 0.8rem 1.1rem; margin: 0.6rem 0; font-size: 0.92rem; max-width: 78%; margin-left: auto; }
.msg-ai { background: white; border: 1px solid var(--mist); border-radius: 16px 16px 16px 4px; padding: 0.8rem 1.1rem; margin: 0.6rem 0; font-size: 0.92rem; max-width: 85%; box-shadow: 0 2px 8px rgba(45,74,62,0.06); line-height: 1.65; white-space: pre-wrap; }

.section-label { font-family: 'DM Serif Display', serif; font-size: 1.2rem; color: var(--forest); border-bottom: 2px solid var(--mist); padding-bottom: 0.3rem; margin: 1.2rem 0 0.8rem; }

.stTextInput > div > div > input {
  border: 2px solid var(--mist) !important; border-radius: 12px !important;
  font-family: 'DM Sans', sans-serif !important; background: white !important;
}
.stTextInput > div > div > input:focus {
  border-color: var(--sage) !important;
  box-shadow: 0 0 0 3px rgba(107,143,113,0.15) !important;
}

.stButton > button {
  background: var(--forest) !important; color: white !important;
  border: none !important; border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
}
.stButton > button:hover { background: var(--sage) !important; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; background: var(--mist); border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; color: var(--sage) !important; }
.stTabs [aria-selected="true"] { background: white !important; color: var(--forest) !important; }

.footer { text-align: center; padding: 1.5rem; color: var(--sage); font-size: 0.8rem; border-top: 1px solid var(--mist); margin-top: 2rem; }
.footer a { color: var(--clay); text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────
for key, default in [("messages", []), ("df", None), ("df_summary", None), ("pending_question", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── API key ────────────────────────────────────────────────────────────
def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return None

API_KEY = get_api_key()

# ── Build data summary ─────────────────────────────────────────────────
def build_df_summary(df):
    summary = {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "nulls": {col: int(df[col].isnull().sum()) for col in df.columns},
        "numeric_stats": {},
        "categorical_stats": {},
        "sample": df.head(5).to_dict(orient="records"),
    }
    for col in df.select_dtypes(include="number").columns:
        summary["numeric_stats"][col] = {
            "mean": round(float(df[col].mean()), 3),
            "median": round(float(df[col].median()), 3),
            "min": round(float(df[col].min()), 3),
            "max": round(float(df[col].max()), 3),
            "std": round(float(df[col].std()), 3),
        }
    for col in df.select_dtypes(include=["object", "category"]).columns:
        vc = df[col].value_counts()
        summary["categorical_stats"][col] = {
            "unique": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in vc.head(5).items()},
        }
    return summary

# ── Call Groq ──────────────────────────────────────────────────────────
def call_groq(messages_history, df_summary, api_key):
    system_prompt = f"""You are an expert data analyst AI. The user uploaded a CSV dataset.

DATASET SUMMARY:
{json.dumps(df_summary, indent=2, default=str)}

Rules:
1. Answer questions clearly and insightfully using only this data.
2. When asked for a chart, include a JSON block exactly like this:
```chart
{{"type": "bar", "x": "column_name", "y": "column_name", "title": "Chart title"}}
```
Supported types: bar, hist, scatter, line
3. Use bullet points for multiple insights. Be concise.
4. NEVER make up data or use columns that don't exist."""

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json={
            "model": "llama-3.1-8b-instant",
            "max_tokens": 1000,
            "messages": [{"role": "system", "content": system_prompt}, *messages_history],
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise Exception(f"Groq API error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"]

# ── Parse response for chart blocks ───────────────────────────────────
def parse_response(text):
    chart_pattern = r"```chart\s*([\s\S]*?)```"
    charts = re.findall(chart_pattern, text)
    clean = re.sub(chart_pattern, "\n📊 *[Chart below]*\n", text)
    return clean, charts

# ── Render chart ───────────────────────────────────────────────────────
def render_chart(df, chart_json_str):
    try:
        spec = json.loads(chart_json_str.strip())
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#FAF7F2')
        ax.set_facecolor('#FAF7F2')
        ctype = spec.get("type", "bar")
        x, y = spec.get("x"), spec.get("y")
        title = spec.get("title", "")

        if ctype == "bar" and x and y and x in df.columns and y in df.columns:
            data = df.groupby(x)[y].mean().reset_index().sort_values(y, ascending=False).head(15)
            ax.bar(data[x].astype(str), data[y], color="#2D4A3E", alpha=0.85, width=0.6)
            ax.set_xlabel(x, fontsize=10, color="#6B8F71")
            ax.set_ylabel(y, fontsize=10, color="#6B8F71")
            plt.xticks(rotation=35, ha='right', fontsize=9)
        elif ctype == "hist" and x and x in df.columns:
            ax.hist(df[x].dropna(), bins=25, color="#2D4A3E", alpha=0.8, edgecolor='white')
            ax.set_xlabel(x, fontsize=10, color="#6B8F71")
            ax.set_ylabel("Count", fontsize=10, color="#6B8F71")
        elif ctype == "scatter" and x and y and x in df.columns and y in df.columns:
            ax.scatter(df[x], df[y], color="#2D4A3E", alpha=0.5, s=20)
            ax.set_xlabel(x, fontsize=10, color="#6B8F71")
            ax.set_ylabel(y, fontsize=10, color="#6B8F71")
        elif ctype == "line" and x and y and x in df.columns and y in df.columns:
            d = df[[x, y]].dropna().sort_values(x)
            ax.plot(d[x], d[y], color="#2D4A3E", linewidth=2)
            ax.set_xlabel(x, fontsize=10, color="#6B8F71")
            ax.set_ylabel(y, fontsize=10, color="#6B8F71")

        ax.set_title(title, fontsize=12, fontweight='bold', color="#2D4A3E", pad=12)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#E8EDE6')
        ax.spines['bottom'].set_color('#E8EDE6')
        ax.tick_params(colors='#6B8F71', labelsize=9)
        plt.tight_layout()
        return fig
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <h1>AI CSV <em>Analyst</em></h1>
  <p>Upload a CSV · Ask questions · Get instant insights & charts</p>
</div>
""", unsafe_allow_html=True)

if not API_KEY:
    st.error("⚠️ No API key found. Add GROQ_API_KEY to your Streamlit secrets (.streamlit/secrets.toml).")
    st.stop()

# ── File upload ────────────────────────────────────────────────────────
uploaded = st.file_uploader("Drop your CSV here", type=["csv"], label_visibility="collapsed")

if uploaded:
    try:
        df = pd.read_csv(uploaded, encoding='latin1')
        st.session_state.df = df
        st.session_state.df_summary = build_df_summary(df)
        st.session_state.messages = []
        st.session_state.pending_question = None
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

# ── Main content ───────────────────────────────────────────────────────
if st.session_state.df is not None:
    df = st.session_state.df

    # Stats row
    num_cols = len(df.select_dtypes(include="number").columns)
    nulls = int(df.isnull().sum().sum())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="label">Rows</div><div class="value">{df.shape[0]:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="label">Columns</div><div class="value">{df.shape[1]}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="label">Numeric</div><div class="value">{num_cols}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card"><div class="label">Missing</div><div class="value">{nulls:,}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["💬 Chat with your data", "🔍 Data preview", "📊 Quick charts"])

    # ── TAB 1: CHAT ────────────────────────────────────────────────────
    with tab1:

        # ── Process any pending question FIRST before rendering ────────
        if st.session_state.pending_question:
            q = st.session_state.pending_question
            st.session_state.pending_question = None
            st.session_state.messages.append({"role": "user", "content": q})
            with st.spinner("🌿 Analysing..."):
                try:
                    api_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    response = call_groq(api_msgs, st.session_state.df_summary, API_KEY)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Error: {e}"})

        # ── Suggestion buttons ─────────────────────────────────────────
        st.markdown("**Try asking:**")
        suggestions = [
            "What are the key insights from this data?",
            "Which columns have missing values?",
            "What patterns do you see?",
            "Show me a chart of the top categories",
        ]
        cols = st.columns(4)
        for i, (col, sug) in enumerate(zip(cols, suggestions)):
            with col:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state.pending_question = sug
                    st.rerun()

        st.markdown("---")

        # ── Chat history ───────────────────────────────────────────────
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

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Input row ──────────────────────────────────────────────────
        col_in, col_btn = st.columns([6, 1])
        with col_in:
            user_input = st.text_input(
                "question",
                placeholder="Ask anything about your data...",
                key="chat_input",
                label_visibility="collapsed",
            )
        with col_btn:
            send = st.button("Send →", use_container_width=True)

        if send and user_input.strip():
            st.session_state.pending_question = user_input.strip()
            st.rerun()

        if st.session_state.messages:
            if st.button("🗑 Clear chat"):
                st.session_state.messages = []
                st.rerun()

    # ── TAB 2: DATA PREVIEW ────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-label">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(50), use_container_width=True, height=380)
        st.markdown('<div class="section-label">Column Info</div>', unsafe_allow_html=True)
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(df[c].dtype) for c in df.columns],
            "Non-null": [int(df[c].notnull().sum()) for c in df.columns],
            "Nulls": [int(df[c].isnull().sum()) for c in df.columns],
            "Unique": [int(df[c].nunique()) for c in df.columns],
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    # ── TAB 3: QUICK CHARTS ────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-label">Explore visually</div>', unsafe_allow_html=True)
        num_columns = list(df.select_dtypes(include="number").columns)
        cat_columns = list(df.select_dtypes(include=["object", "category"]).columns)

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
                ax2.set_title(f"Box plot — {sel_num}", fontsize=11, fontweight='bold', color="#2D4A3E")
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.tick_params(colors='#6B8F71', labelsize=9)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        if cat_columns and num_columns:
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                sel_cat = st.selectbox("Category column", cat_columns, key="qc_cat")
            with c2:
                sel_val = st.selectbox("Value column", num_columns, key="qc_val")
            top_n = df.groupby(sel_cat)[sel_val].mean().sort_values(ascending=False).head(12)
            fig3, ax3 = plt.subplots(figsize=(7, 3.5))
            fig3.patch.set_facecolor('#FAF7F2')
            ax3.set_facecolor('#FAF7F2')
            bar_colors = ["#2D4A3E" if i == 0 else "#6B8F71" if i < 3 else "#A8C5A0" for i in range(len(top_n))]
            ax3.barh(top_n.index.astype(str)[::-1], top_n.values[::-1], color=bar_colors[::-1], alpha=0.85)
            ax3.set_title(f"Avg {sel_val} by {sel_cat}", fontsize=11, fontweight='bold', color="#2D4A3E")
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.tick_params(colors='#6B8F71', labelsize=9)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close()

# ── Empty state ────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="border:2px dashed #A8C5A0;border-radius:16px;padding:2.5rem;text-align:center;background:white;margin:1.5rem 0;">
      <div style="font-size:2.5rem">🌿</div>
      <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;color:#2D4A3E;margin:0.5rem 0;">Upload a CSV to begin</div>
      <div style="color:#6B8F71;font-size:0.9rem;">Sales data · Customer records · Survey results · Any structured CSV</div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Built by <a href="https://mistyvisty.github.io">Preeti Bhardwaj</a> ·
  <a href="https://github.com/mistyvisty/Data-Analytics-Portfolio">GitHub</a>
</div>
""", unsafe_allow_html=True)
