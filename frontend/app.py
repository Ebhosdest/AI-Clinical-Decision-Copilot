"""Streamlit UI for the AI Clinical Decision Copilot."""

from datetime import datetime

from dotenv import load_dotenv
import streamlit as st

load_dotenv(override=True)

st.set_page_config(
    page_title="Clinical Decision Copilot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { color: #1a5276; font-size: 2rem; font-weight: 700; }
    .urgency-emergency { background: #fadbd8; border-left: 4px solid #e74c3c; padding: 10px; border-radius: 4px; }
    .urgency-urgent    { background: #fdebd0; border-left: 4px solid #e67e22; padding: 10px; border-radius: 4px; }
    .urgency-routine   { background: #d5f5e3; border-left: 4px solid #27ae60; padding: 10px; border-radius: 4px; }
    .source-badge { background: #2980b9; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; margin: 2px; display: inline-block; }
    .stTextArea textarea { font-family: 'Segoe UI', sans-serif; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ────────────────────────────────────────────
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "current_response" not in st.session_state:
    st.session_state.current_response = None


# ── In-process pipeline singletons ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_config():
    from src.config import load_config
    try:
        return load_config()
    except ValueError:
        return None


@st.cache_resource(show_spinner=False)
def _get_pipeline():
    from src.rag.pipeline import RAGPipeline
    config = _get_config()
    if config is None:
        return None
    pipeline = RAGPipeline(config)
    try:
        pipeline.load_index()
    except Exception:
        pass  # Index not yet built; will fail gracefully on query
    return pipeline


@st.cache_resource(show_spinner=False)
def _get_graph():
    from src.agents.graph import build_graph
    config = _get_config()
    pipeline = _get_pipeline()
    if config is None or pipeline is None:
        return None
    try:
        return build_graph(config, pipeline)
    except Exception:
        return None


# ── Helper functions ─────────────────────────────────────────────────────────
def check_api_health() -> dict | None:
    config = _get_config()
    if config is None:
        return None
    pipeline = _get_pipeline()
    if pipeline is None:
        return None
    index_loaded = pipeline._vector_store is not None
    index_size = pipeline._vector_store.size if pipeline._vector_store else 0
    return {
        "index_loaded": index_loaded,
        "index_size": index_size,
        "model": config.chat_model,
    }


def submit_query(query: str, use_agents: bool, patient_info: dict | None) -> dict | None:
    try:
        if use_agents:
            graph = _get_graph()
            if graph is None:
                st.error("Agent graph could not be initialised. Check your OPENAI_API_KEY.")
                return None
            from src.agents.state import AgentState
            initial: AgentState = {"query": query}
            if patient_info:
                initial["patient_info"] = patient_info  # type: ignore[assignment]
            state = graph.invoke(initial)
            diagnoses = [d for d in state.get("diagnoses", []) if isinstance(d, dict)]
            return {
                "query": query,
                "response": state.get("response", ""),
                "sources": state.get("sources", []),
                "urgency": state.get("urgency"),
                "diagnoses": diagnoses,
                "num_chunks_retrieved": len(state.get("raw_results", [])),
                "agent_path": ["triage", "retrieval", "reasoning", "response"],
            }
        else:
            pipeline = _get_pipeline()
            if pipeline is None:
                st.error("Pipeline could not be initialised. Check your OPENAI_API_KEY.")
                return None
            result = pipeline.query(query)
            return {
                "query": query,
                "response": result["response"],
                "sources": result["sources"],
                "urgency": None,
                "diagnoses": [],
                "num_chunks_retrieved": result["num_chunks_retrieved"],
                "agent_path": None,
            }
    except Exception as e:
        st.error(f"Query error: {e}")
    return None


def stream_query(query: str) -> str:
    """Stream tokens directly from the RAG pipeline generator."""
    full_response = ""
    placeholder = st.empty()
    try:
        pipeline = _get_pipeline()
        if pipeline is None:
            st.error("Pipeline could not be initialised. Check your OPENAI_API_KEY.")
            return ""
        for token in pipeline.query_stream(query):
            full_response += token
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)
    except Exception as e:
        st.error(f"Streaming error: {e}")
    return full_response


def urgency_badge(urgency: str) -> str:
    colours = {
        "emergency": "#e74c3c",
        "urgent": "#e67e22",
        "routine": "#27ae60",
        "information_request": "#2980b9",
    }
    colour = colours.get(urgency, "#95a5a6")
    return f'<span style="background:{colour};color:white;padding:3px 10px;border-radius:12px;font-size:0.85rem;font-weight:600">{urgency.upper()}</span>'


def build_index() -> dict:
    try:
        config = _get_config()
        if config is None:
            return {"status": "error", "message": "OPENAI_API_KEY not configured.", "chunks_indexed": 0}
        from src.rag.pipeline import RAGPipeline
        pipeline = RAGPipeline(config)
        pipeline.build_index()
        # Clear cached singletons so the next call loads the fresh index
        _get_pipeline.clear()
        _get_graph.clear()
        chunks = pipeline._vector_store.size if pipeline._vector_store else 0
        return {"status": "success", "message": "Index built.", "chunks_indexed": chunks}
    except Exception as e:
        return {"status": "error", "message": str(e), "chunks_indexed": 0}


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Clinical Copilot")
    st.markdown("---")

    health = check_api_health()
    if health:
        status_colour = "🟢" if health.get("index_loaded") else "🟡"
        st.markdown(f"{status_colour} **Mode:** Standalone")
        st.markdown(f"📚 **Index:** {health.get('index_size', 0)} vectors")
        st.markdown(f"🤖 **Model:** `{health.get('model', 'N/A')}`")
    else:
        st.markdown("🔴 **Status:** Not configured")
        st.info("Set OPENAI_API_KEY in your .env file.")

    st.markdown("---")

    with st.expander("⚙️ Settings"):
        use_agents = st.toggle("Multi-Agent Pipeline", value=True, help="Use full LangGraph pipeline")
        use_streaming = st.toggle("Stream Response", value=False, help="Stream tokens (simple RAG only)")

    st.markdown("---")

    with st.expander("🏗️ Index Management"):
        if st.button("Rebuild Index", type="secondary"):
            with st.spinner("Building index..."):
                result = build_index()
            if result.get("status") == "success":
                st.success(f"✅ {result.get('chunks_indexed', 0)} chunks indexed")
            else:
                st.error(result.get("message", "Build failed"))

    st.markdown("---")
    st.markdown("### 📋 Query History")
    if st.session_state.query_history:
        for i, item in enumerate(reversed(st.session_state.query_history[-10:])):
            if st.button(f"↩ {item['query'][:40]}...", key=f"hist_{i}"):
                st.session_state.current_response = item
    else:
        st.caption("No queries yet.")


# ── Main Panel ────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">🩺 AI Clinical Decision Copilot</h1>', unsafe_allow_html=True)
st.markdown("*Evidence-based clinical decision support — for healthcare professionals only*")
st.markdown("---")

col_query, col_patient = st.columns([3, 1])

with col_query:
    query = st.text_area(
        "Clinical Query",
        placeholder="Describe the clinical scenario or ask a guideline question...\n\nExample: 55-year-old male with acute chest pain radiating to left arm, diaphoresis, ECG shows ST elevation in leads II, III, aVF. What is the management?",
        height=150,
        label_visibility="collapsed",
    )

with col_patient:
    st.markdown("**Patient Info (optional)**")
    age = st.number_input("Age", min_value=0, max_value=120, value=0)
    sex = st.selectbox("Sex", ["", "male", "female", "other"])
    comorbidities = st.text_input("Comorbidities", placeholder="HTN, DM, ...")
    medications = st.text_input("Medications", placeholder="aspirin, metformin...")

patient_info = {}
if age > 0:
    patient_info["age"] = age
if sex:
    patient_info["sex"] = sex
if comorbidities:
    patient_info["comorbidities"] = [c.strip() for c in comorbidities.split(",")]
if medications:
    patient_info["medications"] = [m.strip() for m in medications.split(",")]

submit_col, clear_col, _ = st.columns([1, 1, 4])
with submit_col:
    submit = st.button("🔍 Analyse", type="primary", disabled=not query.strip())
with clear_col:
    if st.button("🗑️ Clear"):
        st.session_state.current_response = None
        st.rerun()

# ── Query Execution ───────────────────────────────────────────────────────────
if submit and query.strip():
    if not health:
        st.error("Pipeline not configured. Set OPENAI_API_KEY in your .env file.")
    else:
        with st.spinner("Analysing clinical query..."):
            if use_streaming and not use_agents:
                st.markdown("### Response")
                response_text = stream_query(query)
                result = {
                    "query": query,
                    "response": response_text,
                    "sources": [],
                    "urgency": "unknown",
                    "diagnoses": [],
                    "num_chunks_retrieved": 0,
                }
            else:
                result = submit_query(query, use_agents, patient_info if patient_info else None)

        if result:
            st.session_state.current_response = result
            st.session_state.query_history.append({
                "query": query[:80],
                "timestamp": datetime.now().isoformat(),
                **result,
            })

# ── Response Display ──────────────────────────────────────────────────────────
if st.session_state.current_response:
    result = st.session_state.current_response
    st.markdown("---")

    # Urgency banner
    urgency = result.get("urgency")
    if urgency and urgency != "unknown":
        urgency_class = f"urgency-{urgency}" if urgency in ("emergency", "urgent", "routine") else "urgency-routine"
        st.markdown(
            f'<div class="{urgency_class}"><strong>Triage Classification:</strong> {urgency_badge(urgency)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    # Main response
    tab_response, tab_diagnoses, tab_sources, tab_debug = st.tabs(
        ["📄 Response", "🔬 Diagnoses", "📚 Sources", "🔧 Debug"]
    )

    with tab_response:
        st.markdown(result.get("response", "*No response generated.*"))

    with tab_diagnoses:
        diagnoses = result.get("diagnoses", [])
        if diagnoses:
            for d in diagnoses:
                likelihood_colour = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    d.get("likelihood", "").lower(), "⚪"
                )
                with st.expander(
                    f"{likelihood_colour} **{d.get('diagnosis', 'Unknown')}** — {d.get('likelihood', '').title()} Likelihood"
                ):
                    st.markdown(f"**Reasoning:** {d.get('reasoning', 'N/A')}")
                    flags = d.get("red_flags", [])
                    if flags:
                        st.warning("**Red Flags:** " + " | ".join(flags))
        else:
            st.info("Differential diagnoses not available for this query type.")

    with tab_sources:
        sources = result.get("sources", [])
        chunks = result.get("num_chunks_retrieved", 0)
        st.markdown(f"**Guideline chunks retrieved:** {chunks}")
        if sources:
            st.markdown("**Sources cited:**")
            for s in sources:
                st.markdown(f'<span class="source-badge">📖 {s}</span>', unsafe_allow_html=True)
        else:
            st.info("No sources available.")

    with tab_debug:
        agent_path = result.get("agent_path")
        if agent_path:
            st.markdown(f"**Agent path:** {' → '.join(agent_path)}")
        st.json({k: v for k, v in result.items() if k != "response"})


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>⚠️ For qualified healthcare professionals only. Not a substitute for clinical judgement.</small>",
    unsafe_allow_html=True,
)
