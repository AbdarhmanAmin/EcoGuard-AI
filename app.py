"""
EcoGuard AI - Water Intelligence Platform
==========================================
تطبيق Streamlit احترافي يجمع 3 أنظمة:
1. مساعد ذكي (RAG) للإجابة على أسئلة جودة المياه من مستندات PDF.
2. نموذج توقع تسرب المياه (SVM).
3. نموذج توقع تلوث المياه (XGBoost).

هيكل المشروع المتوقع (جذر الريبو):
EcoGuard-AI/
├── app.py                              <- هذا الملف
├── requirements.txt
├── Model 1 _ Water Leakage/
│   ├── Water_leakage_model.pkl
│   └── scaler_leakage.pkl
├── Model 2 _Water Contamination/
│   ├── water_contamination_model.pkl
│   └── scaler_contamination.pkl
└── Model 3 _ RAG/
    ├── documents/
    └── vector_db/
"""

import os

import joblib
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# ----------------------------------------------------------------------------
# إعداد الصفحة العام
# ----------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="EcoGuard AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# مسارات الفولدرات (مطابقة تمامًا لأسماء الفولدرات على GitHub)
MODEL1_DIR = "Model 1 _ Water Leakage"
MODEL2_DIR = "Model 2 _Water Contamination"
MODEL3_DIR = "Model 3 _ RAG"

MODEL1_PATH = os.path.join(MODEL1_DIR, "Water_leakage_model.pkl")
SCALER1_PATH = os.path.join(MODEL1_DIR, "scaler_leakage.pkl")

MODEL2_PATH = os.path.join(MODEL2_DIR, "water_contamination_model.pkl")
SCALER2_PATH = os.path.join(MODEL2_DIR, "scaler_contamination.pkl")

PDF_FOLDER = os.path.join(MODEL3_DIR, "documents")
VECTOR_DB_DIR = os.path.join(MODEL3_DIR, "vector_db")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ----------------------------------------------------------------------------
# تنسيق عام (CSS) - نظام تصميم استارت أب احترافي
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Inter:wght@400;600;700&display=swap');

    :root {
        --eg-primary: #06b6d4;
        --eg-primary-dark: #0891b2;
        --eg-accent: #22d3ee;
        --eg-navy: #0b1f2e;
        --eg-navy-light: #12293b;
        --eg-success: #22c55e;
        --eg-danger: #ef4444;
        --eg-text-soft: rgba(255,255,255,0.72);
    }

    html, body, [class*="css"] { font-family: 'Cairo', 'Inter', sans-serif; }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(6,182,212,0.10) 0%, transparent 45%),
            radial-gradient(circle at 100% 20%, rgba(34,211,238,0.08) 0%, transparent 40%),
            var(--eg-navy);
    }

    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }

    /* ---------- Hero ---------- */
    .eg-hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(120deg, #063d4d 0%, #0891b2 55%, #06b6d4 100%);
        padding: 2.8rem 2.6rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 50px -20px rgba(6, 182, 212, 0.45);
    }
    .eg-hero::after {
        content: "";
        position: absolute; top: -60px; right: -60px;
        width: 220px; height: 220px; border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 70%);
    }
    .eg-badge {
        display: inline-block;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-size: 0.82rem;
        margin-bottom: 0.9rem;
        letter-spacing: 0.3px;
    }
    .eg-hero h1 { margin: 0; font-size: 2.4rem; font-weight: 800; }
    .eg-hero p { margin: 0.6rem 0 0 0; opacity: 0.94; font-size: 1.08rem; max-width: 640px; }

    /* ---------- Stat pills on hero ---------- */
    .eg-stats-row { display: flex; gap: 1rem; margin-top: 1.6rem; flex-wrap: wrap; }
    .eg-stat {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.22);
        border-radius: 14px;
        padding: 0.7rem 1.1rem;
        min-width: 130px;
    }
    .eg-stat .num { font-size: 1.35rem; font-weight: 800; }
    .eg-stat .lbl { font-size: 0.8rem; opacity: 0.85; }

    /* ---------- Feature cards ---------- */
    .eg-card {
        background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 1.6rem 1.5rem;
        height: 100%;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .eg-card:hover {
        transform: translateY(-6px);
        border-color: rgba(6,182,212,0.55);
        box-shadow: 0 18px 35px -18px rgba(6,182,212,0.55);
    }
    .eg-card .icon-circle {
        width: 52px; height: 52px; border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem;
        background: linear-gradient(135deg, var(--eg-primary), var(--eg-accent));
        margin-bottom: 0.9rem;
        box-shadow: 0 10px 20px -8px rgba(6,182,212,0.65);
    }
    .eg-card h3 { margin: 0 0 0.5rem 0; color: white; font-size: 1.12rem; }
    .eg-card p { margin: 0; color: var(--eg-text-soft); font-size: 0.93rem; line-height: 1.6; }

    /* ---------- Result banners ---------- */
    .eg-result-safe, .eg-result-danger {
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        font-size: 1.1rem;
        display: flex; align-items: center; gap: 0.8rem;
        margin-top: 0.6rem;
    }
    .eg-result-safe {
        background: linear-gradient(120deg, rgba(34,197,94,0.16), rgba(34,197,94,0.05));
        border: 1px solid rgba(34, 197, 94, 0.45);
        color: #d1fae5;
    }
    .eg-result-danger {
        background: linear-gradient(120deg, rgba(239,68,68,0.18), rgba(239,68,68,0.05));
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #fee2e2;
    }
    .eg-result-safe b, .eg-result-danger b { color: white; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1f2e 0%, #0d2536 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    .eg-logo-wrap { display: flex; align-items: center; gap: 0.7rem; padding: 0.4rem 0 0.2rem 0; }
    .eg-logo-circle {
        width: 42px; height: 42px; border-radius: 12px;
        background: linear-gradient(135deg, var(--eg-primary), var(--eg-accent));
        display: flex; align-items: center; justify-content: center; font-size: 1.3rem;
        box-shadow: 0 8px 18px -6px rgba(6,182,212,0.6);
    }
    .eg-logo-text { font-weight: 800; font-size: 1.15rem; color: white; line-height: 1.1; }
    .eg-logo-sub { font-size: 0.75rem; color: var(--eg-text-soft); }

    section[data-testid="stSidebar"] .stRadio > label { display: none; }
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.5rem;
        width: 100%;
        transition: all 0.2s ease;
        font-size: 0.98rem;
        color: var(--eg-text-soft);
    }
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
        border-color: rgba(6,182,212,0.5);
        background: rgba(6,182,212,0.08);
    }

    /* ---------- Buttons ---------- */
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(135deg, var(--eg-primary-dark), var(--eg-accent));
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.6rem 1.2rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 10px 22px -10px rgba(6,182,212,0.7);
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 28px -10px rgba(6,182,212,0.85);
    }

    /* ---------- Inputs ---------- */
    div[data-baseweb="input"], div[data-baseweb="base-input"] {
        border-radius: 10px !important;
    }

    .eg-footer {
        text-align: center;
        color: var(--eg-text-soft);
        font-size: 0.82rem;
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# التنقل بين الصفحات
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="eg-logo-wrap">
            <div class="eg-logo-circle">🌊</div>
            <div>
                <div class="eg-logo-text">EcoGuard AI</div>
                <div class="eg-logo-sub">Water Intelligence Platform</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "💧 Water Leakage Prediction",
            "🧪 Water Quality Prediction",
            "🤖 Water Assistant (RAG)",
        ],
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="eg-footer">EcoGuard AI © 2026<br>Powered by Groq + LangChain</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# الصفحة 1: الرئيسية
# ----------------------------------------------------------------------------
def render_home():
    st.markdown(
        """
        <div class="eg-hero">
            <span class="eg-badge">⚡ Integrated AI Platform</span>
            <h1>🌊 EcoGuard AI</h1>
            <p>Three AI systems in one place to monitor water networks,
            detect leaks, and assess water quality in real time — built on
            state-of-the-art machine learning and language models.</p>
            <div class="eg-stats-row">
                <div class="eg-stat"><div class="num">3</div><div class="lbl">AI Systems</div></div>
                <div class="eg-stat"><div class="num">96%+</div><div class="lbl">Leak Detection Accuracy</div></div>
                <div class="eg-stat"><div class="num">99%+</div><div class="lbl">Contamination Detection Accuracy</div></div>
                <div class="eg-stat"><div class="num">24/7</div><div class="lbl">Real-time Response</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🚀 Available Systems")
    st.write("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="eg-card">
                <div class="icon-circle">💧</div>
                <h3>Water Leakage Prediction</h3>
                <p>An SVM model analyzes pressure, flow rate, vibration, and rotation
                data to detect early signs of leaks in the pipe network before they escalate.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="eg-card">
                <div class="icon-circle">🧪</div>
                <h3>Water Quality Prediction</h3>
                <p>An XGBoost model assesses water safety (Safe / Unsafe) based on
                pH, turbidity, total dissolved solids, and temperature.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="eg-card">
                <div class="icon-circle">🤖</div>
                <h3>Water Assistant</h3>
                <p>A RAG-powered AI assistant that answers your questions based on
                official water quality standards and documents, accurately and reliably.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.info("💡 Use the sidebar to navigate between the three systems and get started.")

    st.markdown(
        '<div class="eg-footer">EcoGuard AI © 2026 — Built with Streamlit · LangChain · Groq</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# الصفحة 2: توقع تسرب المياه
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_leakage_assets():
    model = joblib.load(MODEL1_PATH)
    scaler = joblib.load(SCALER1_PATH)
    return model, scaler


def render_leakage_page():
    st.markdown(
        """
        <div class="eg-hero" style="padding:1.8rem 2rem;">
            <span class="eg-badge">SVM Classifier</span>
            <h1 style="font-size:1.7rem;">💧 Water Leakage Prediction</h1>
            <p>Enter live sensor readings to predict the likelihood of a leak in the pipe network.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not os.path.exists(MODEL1_PATH) or not os.path.exists(SCALER1_PATH):
        st.error(f"Model files not found inside `{MODEL1_DIR}/`.")
        return

    model, scaler = load_leakage_assets()

    with st.form("leakage_form"):
        c1, c2 = st.columns(2)
        with c1:
            pressure = st.number_input("Pressure", min_value=0.0, max_value=200.0, value=60.0)
            flow_rate = st.number_input("Flow Rate", min_value=0.0, max_value=300.0, value=80.0)
            temperature = st.number_input("Temperature", min_value=0.0, max_value=200.0, value=100.0)
            vibration = st.number_input("Vibration", min_value=0.0, max_value=20.0, value=3.0)
        with c2:
            rpm = st.number_input("RPM", min_value=0.0, max_value=6000.0, value=2000.0)
            operational_hours = st.number_input("Operational Hours", min_value=0, max_value=20000, value=5500)
            latitude = st.number_input("Latitude", value=25.19, format="%.6f")
            longitude = st.number_input("Longitude", value=55.25, format="%.6f")

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        features = np.array([[
            pressure, flow_rate, temperature, vibration,
            rpm, operational_hours, latitude, longitude,
        ]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        if prediction == 1:
            st.markdown(
                '<div class="eg-result-danger"><span style="font-size:1.8rem;">⚠️</span>'
                '<div><b>Warning: High Leak Probability</b><br>'
                'The model predicts a water leak based on the provided readings.'
                ' A field inspection by a technical team is recommended.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="eg-result-safe"><span style="font-size:1.8rem;">✅</span>'
                '<div><b>Network is Healthy</b><br>'
                'No signs of leakage detected. The network is operating within normal range.</div></div>',
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------------
# الصفحة 3: توقع جودة المياه
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_contamination_assets():
    model = joblib.load(MODEL2_PATH)
    scaler = joblib.load(SCALER2_PATH)
    return model, scaler


def render_contamination_page():
    st.markdown(
        """
        <div class="eg-hero" style="padding:1.8rem 2rem;">
            <span class="eg-badge">XGBoost Classifier</span>
            <h1 style="font-size:1.7rem;">🧪 Water Quality Prediction</h1>
            <p>Enter chemical water measurements to assess its safety for use.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not os.path.exists(MODEL2_PATH) or not os.path.exists(SCALER2_PATH):
        st.error(f"Model files not found inside `{MODEL2_DIR}/`.")
        return

    model, scaler = load_contamination_assets()

    with st.form("contamination_form"):
        c1, c2 = st.columns(2)
        with c1:
            ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.4, step=0.01)
            tds_ppm = st.number_input("Total Dissolved Solids (TDS ppm)", min_value=0.0, max_value=5000.0, value=690.0)
        with c2:
            turbidity_ntu = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=200.0, value=11.0)
            temperature_c = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0)

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        features = np.array([[ph, tds_ppm, turbidity_ntu, temperature_c]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        if prediction == 1:
            st.markdown(
                '<div class="eg-result-danger"><span style="font-size:1.8rem;">⚠️</span>'
                '<div><b>Unsafe Water</b><br>'
                'The model classifies this sample as unsafe for use '
                'based on the provided measurements.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="eg-result-safe"><span style="font-size:1.8rem;">✅</span>'
                '<div><b>Safe Water</b><br>'
                'The provided measurements are within safe usage ranges.</div></div>',
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------------
# الصفحة 4: مساعد RAG
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str):
    from langchain_groq import ChatGroq
    return ChatGroq(model=LLM_MODEL_NAME, temperature=0, groq_api_key=api_key)


def vector_db_exists() -> bool:
    return os.path.isdir(VECTOR_DB_DIR) and len(os.listdir(VECTOR_DB_DIR)) > 0


def build_vector_db():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    documents = []
    loaded_files = []
    if os.path.isdir(PDF_FOLDER):
        for file in sorted(os.listdir(PDF_FOLDER)):
            if file.lower().endswith(".pdf"):
                path = os.path.join(PDF_FOLDER, file)
                docs = PyPDFLoader(path).load()
                documents.extend(docs)
                loaded_files.append((file, len(docs)))

    if not documents:
        return None, loaded_files, 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)

    embedding_model = get_embedding_model()
    vector_db = Chroma.from_documents(
        documents=chunks, embedding=embedding_model, persist_directory=VECTOR_DB_DIR
    )
    return vector_db, loaded_files, len(chunks)


@st.cache_resource(show_spinner=False)
def load_existing_vector_db():
    from langchain_community.vectorstores import Chroma
    embedding_model = get_embedding_model()
    return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embedding_model)


RAG_PROMPT_TEMPLATE = """You are a water quality assistant.

Use the following context to help answer the question if it is relevant.
If the context does not contain the answer, answer normally using your own
general knowledge instead of saying you don't have enough information.

Context:
{context}

Question:
{question}

Answer:"""


def rag_answer(llm, retriever, question: str):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)
    return response.content, docs


def render_rag_page():
    st.markdown(
        """
        <div class="eg-hero" style="padding:1.8rem 2rem;">
            <span class="eg-badge">RAG · Groq · Llama 3.3</span>
            <h1 style="font-size:1.7rem;">🤖 Water Assistant</h1>
            <p>Ask any question about water quality and the assistant will answer based on official reference documents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### ⚙️ Assistant Settings")
        env_api_key = os.getenv("GROQ_API_KEY", "")
        api_key = st.text_input(
            "Groq API Key", value=env_api_key, type="password",
            help="Completely free from console.groq.com",
        )
        rebuild = st.button("🔄 Rebuild Knowledge Base", use_container_width=True)

    if rebuild:
        with st.spinner("Rebuilding knowledge base..."):
            load_existing_vector_db.clear()
            vector_db, loaded_files, n_chunks = build_vector_db()
        if vector_db is None:
            st.error(f"No PDF files found inside `{PDF_FOLDER}`.")
            return
        st.success(f"Knowledge base built: {len(loaded_files)} files, {n_chunks} chunks.")
    elif not vector_db_exists():
        st.info("Building the knowledge base for the first time from the documents...")
        with st.spinner("Processing PDF files..."):
            vector_db, loaded_files, n_chunks = build_vector_db()
        if vector_db is None:
            st.warning(f"Add PDF files inside `{PDF_FOLDER}` then reload the page.")
            return
        st.success(f"Knowledge base built: {len(loaded_files)} files, {n_chunks} chunks.")
    else:
        vector_db = load_existing_vector_db()

    retriever = vector_db.as_retriever()

    if not api_key:
        st.warning("Enter your Groq API Key in the sidebar to continue.")
        return

    llm = get_llm(api_key)

    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []

    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- {s}")

    question = st.chat_input("Type your question here...")
    if question:
        st.session_state.rag_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating an answer..."):
                try:
                    answer, docs = rag_answer(llm, retriever, question)
                except Exception as e:
                    answer, docs = f"An error occurred: {e}", []
            st.markdown(answer)
            sources = []
            if docs:
                with st.expander("Sources"):
                    for d in docs:
                        src = os.path.basename(d.metadata.get("source", "unknown"))
                        page = d.metadata.get("page", "?")
                        label = f"{src} - page {page}"
                        sources.append(label)
                        st.markdown(f"- {label}")

        st.session_state.rag_messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


# ----------------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------------
if page == "🏠 Home":
    render_home()
elif page == "💧 Water Leakage Prediction":
    render_leakage_page()
elif page == "🧪 Water Quality Prediction":
    render_contamination_page()
elif page == "🤖 Water Assistant (RAG)":
    render_rag_page()
