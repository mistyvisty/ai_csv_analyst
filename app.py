import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, re, requests, warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI CSV Analyst · Preeti Bhardwaj", page_icon="🌿", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{--cream:#FAF7F2;--forest:#2D4A3E;--sage:#6B8F71;--moss:#A8C5A0;--clay:#C4875A;--mist:#E8EDE6;}
html,body,[class*="css"],.stApp{background-color:var(--cream)!important;font-family:'DM Sans',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1rem!important;}
.hero{padding:1.5rem 0 1rem;text-align:center;}
.hero h1{font-family:'DM Serif Display',serif;font-size:2.6rem;color:var(--forest);margin:0;}
.hero h1 em{color:var(--clay);font-style:italic;}
.hero p{font-size:0.95rem;color:var(--sage);margin:0.3rem 0 0;}
.stat-card{background:white;border-radius:12px;padding:1rem 1.4rem;border-left:4px solid var(--sage);box-shadow:0 2px 8px rgba(45,74,62,0.06);}
.stat-card .label{font-size:0.72rem;color:var(--sage);text-transform:uppercase;letter-spacing:0.08em;font-weight:600;}
.stat-card .value{font-size:1.5rem;font-family:'DM Serif Display',serif;color:var(--forest);}
.stButton>button{background:var(--forest)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:500!important;}
.stButton>button:hover{background:var(--sage)!important;}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:var(--mist);border-radius:12px;padding:4px;}
.stTabs [data-baseweb="tab"]{border-radius:8px!important;font-weight:500!important;color:var(--sage)!important;}
.stTabs [aria-selected="true"]{background:white!important;color:var(--forest)!important;}
.section-label{font-family:'DM Serif Display',serif;font-size:1.2rem;color:var(--forest);border-bottom:2px solid var(--mist);padding-bottom:0.3rem;margin:1.2rem 0 0.8rem;}
.footer{text-align:center;padding:1.5rem;color:var(--sage);font-size:0.8rem;border-top:1px solid var(--mist);margin-top:2rem;}
.footer a{color:var(--clay);text-decoration:none;}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "df_summary" not in st.session_state:
    st.session_state.df_summary = None

def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return None

API_KEY = get_api_key()

def build_df_summary(df):
    summary = {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "nulls": {c: int(df[c].isnull().sum()) for c in df.columns},
        "numeric_stats": {},
        "categorical_stats": {},
        "sample": df.head(3).to_dict(orient="records"),
    }
    for c in df.select_dtypes(include="number").columns:
        summary["numeric_stats"][c] = {"mean": round(float(df[c].mean()),2), "min": round(float(df[c].min()),2), "max": round(float(df[c].max()),2)}
    for c in df.select_dtypes(include=["object","category"]).columns:
        summary["categorical_stats"][c] = {"unique": int(df[c].nunique()), "top": df[c].value_counts().head(3).to_dict()}
    return summary

def call_groq(api_key, df_summary, messages):
    system = f"""You are an expert data analyst. The user uploaded a CSV.

DATASET INFO:
{json.dumps(df_summary, indent=2, default=str)}

Rules:
1. Answer only from this data. Be clear and concise.
2. For charts, output a JSON block like:
```chart
{{"type":"bar","x":"col_name","y":"col_name","title":"My Chart"}}
```
Types: bar, hist, scatter, line
3. Never invent columns or data."""

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "llama-3.1-8b-instant", "max_tokens": 800,
              "messages": [{"role": "system", "content": system}] + messages},
        timeout=30,
    )
    if r.status_code != 200:
        raise Exception(f"Groq {r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"]

def render_chart(df, spec_str):
    try:
        spec = json.loads(spec_str.strip())
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor('#FAF7F2')
        ax.set_facecolor('#FAF7F2')
        t, x, y = spec.get("type","bar"), spec.get("x"), spec.get("y")
        title = spec.get("title","")
        if t == "bar" and x and y and x in df.columns and y in df.columns:
            d = df.groupby(x)[y].mean().sort_values(ascending=False).head(12)
            ax.bar(d.index.astype(str), d.values, color="#2D4A3E", alpha=0.85)
            plt.xticks(rotation=35, ha='right', fontsize=9)
        elif t == "hist" and x and x in df.columns:
            ax.hist(df[x].dropna(), bins=20, color="#2D4A3E", alpha=0.8, edgecolor='white')
        elif t == "scatter" and x and y and x in df.columns and y in df.columns:
            ax.scatter(df[x], df[y], color="#2D4A3E", alpha=0.5, s=15)
        elif t == "line" and x and y and x in df.columns and y in df.columns:
            d = df[[x,y]].dropna().sort_values(x)
            ax.plot(d[x], d[y], color="#2D4A3E", lw=2)
        ax.set_title(title, fontsize=11, fontweight='bold', color="#2D4A3E")
        for s in ['top','right']: ax.spines[s].set_visible(False)
        ax.tick_params(colors='#6B8F71', labelsize=9)
        plt.tight_layout()
        return fig
    except:
        return None

def display_response(df, content):
    """Display AI response text + any charts."""
    chart_pattern = r"```chart\s*([\s\S]*?)```"
    charts = re.findall(chart_pattern, content)
    clean = re.sub(chart_pattern, "\n📊 *(chart below)*\n", content)
    st.markdown(clean)
    for c in charts:
        fig = render_chart(df, c)
        if fig:
            st.pyplot(fig, use_container_width=True)
            plt.close()

def ask(question):
    """Send question to Groq, save both to session state."""
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        reply = call_groq(API_KEY, st.session_state.df_summary, st.session_state.messages)
    except Exception as e:
        reply = f"⚠️ {e}"
    st.session_state.messages.append({"role": "assistant", "content": reply})

# ══════════════════════════════════════════
# PAGE
# ══════════════════════════════════════════

st.markdown('<div class="hero"><h1>AI CSV <em>Analyst</em></h1><p>Upload a CSV · Ask questions · Get instant insights & charts</p></div>', unsafe_allow_html=True)

if not API_KEY:
    st.error("⚠️ Add GROQ_API_KEY to Streamlit secrets.")
    st.stop()

uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
if uploaded:
    try:
        df_new = pd.read_csv(uploaded, encoding='latin1')
        st.session_state.df = df_new
        st.session_state.df_summary = build_df_summary(df_new)
        st.session_state.messages = []
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

if st.session_state.df is not None:
    df = st.session_state.df
    num_cols = len(df.select_dtypes(include="number").columns)
    nulls = int(df.isnull().sum().sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="stat-card"><div class="label">Rows</div><div class="value">{df.shape[0]:,}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="stat-card"><div class="label">Columns</div><div class="value">{df.shape[1]}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="stat-card"><div class="label">Numeric</div><div class="value">{num_cols}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="stat-card"><div class="label">Missing</div><div class="value">{nulls:,}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Data preview", "📊 Quick charts"])

    with tab1:
        # Suggestion buttons
        st.markdown("**Try asking:**")
        sugs = ["What are the key insights?", "Any missing values?", "What patterns do you see?", "Show chart of top categories"]
        bc = st.columns(4)
        clicked = None
        for i, (col, s) in enumerate(zip(bc, sugs)):
            with col:
                if st.button(s, key=f"s{i}", use_container_width=True):
                    clicked = s
        if clicked:
            with st.spinner("🌿 Analysing..."):
                ask(clicked)

        st.divider()

        # Chat history
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                if m["role"] == "user":
                    st.markdown(m["content"])
                else:
                    display_response(df, m["content"])

        # Input — outside any container, directly in tab
        prompt = st.chat_input("Ask anything about your data...")
        if prompt:
            with st.spinner("🌿 Analysing..."):
                ask(prompt)
            st.rerun()

        if st.session_state.messages:
            if st.button("🗑 Clear chat", key="clear"):
                st.session_state.messages = []
                st.rerun()

    with tab2:
        st.markdown('<div class="section-label">Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(50), use_container_width=True, height=360)
        st.markdown('<div class="section-label">Column Info</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Column": df.columns,
            "Type": [str(df[c].dtype) for c in df.columns],
            "Non-null": [int(df[c].notnull().sum()) for c in df.columns],
            "Nulls": [int(df[c].isnull().sum()) for c in df.columns],
            "Unique": [int(df[c].nunique()) for c in df.columns],
        }), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown('<div class="section-label">Explore visually</div>', unsafe_allow_html=True)
        num_cols_list = list(df.select_dtypes(include="number").columns)
        cat_cols_list = list(df.select_dtypes(include=["object","category"]).columns)

        if num_cols_list:
            c1, c2 = st.columns(2)
            with c1: sel_num = st.selectbox("Numeric column", num_cols_list, key="qn")
            with c2: ctype = st.selectbox("Chart", ["Histogram","Box plot"], key="qt")
            fig2, ax2 = plt.subplots(figsize=(7,3.5))
            fig2.patch.set_facecolor('#FAF7F2'); ax2.set_facecolor('#FAF7F2')
            if ctype == "Histogram":
                ax2.hist(df[sel_num].dropna(), bins=25, color="#2D4A3E", alpha=0.8, edgecolor='white')
                ax2.set_title(f"Distribution of {sel_num}", fontsize=11, fontweight='bold', color="#2D4A3E")
            else:
                ax2.boxplot(df[sel_num].dropna(), patch_artist=True,
                            boxprops=dict(facecolor="#A8C5A0",alpha=0.8),
                            medianprops=dict(color="#2D4A3E",linewidth=2))
                ax2.set_title(f"Box plot — {sel_num}", fontsize=11, fontweight='bold', color="#2D4A3E")
            for s in ['top','right']: ax2.spines[s].set_visible(False)
            ax2.tick_params(colors='#6B8F71', labelsize=9)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True); plt.close()

        if cat_cols_list and num_cols_list:
            st.divider()
            c1, c2 = st.columns(2)
            with c1: sel_cat = st.selectbox("Category", cat_cols_list, key="qc")
            with c2: sel_val = st.selectbox("Value", num_cols_list, key="qv")
            top_n = df.groupby(sel_cat)[sel_val].mean().sort_values(ascending=False).head(12)
            fig3, ax3 = plt.subplots(figsize=(7,3.5))
            fig3.patch.set_facecolor('#FAF7F2'); ax3.set_facecolor('#FAF7F2')
            bcolors = ["#2D4A3E" if i==0 else "#6B8F71" if i<3 else "#A8C5A0" for i in range(len(top_n))]
            ax3.barh(top_n.index.astype(str)[::-1], top_n.values[::-1], color=bcolors[::-1], alpha=0.85)
            ax3.set_title(f"Avg {sel_val} by {sel_cat}", fontsize=11, fontweight='bold', color="#2D4A3E")
            for s in ['top','right']: ax3.spines[s].set_visible(False)
            ax3.tick_params(colors='#6B8F71', labelsize=9)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True); plt.close()

else:
    st.markdown("""
    <div style="border:2px dashed #A8C5A0;border-radius:16px;padding:2.5rem;text-align:center;background:white;margin:1.5rem 0;">
      <div style="font-size:2.5rem">🌿</div>
      <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;color:#2D4A3E;margin:0.5rem 0;">Upload a CSV to begin</div>
      <div style="color:#6B8F71;font-size:0.9rem;">Sales data · Customer records · Survey results · Any structured CSV</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer">Built by <a href="https://mistyvisty.github.io">Preeti Bhardwaj</a> · <a href="https://github.com/mistyvisty/Data-Analytics-Portfolio">GitHub</a></div>', unsafe_allow_html=True)
