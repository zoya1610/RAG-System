from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv
import singlestoredb as s2

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from loaders import uploaded_files_to_documents
from prompts import SYSTEM_PROMPT
from rag_singlestore import RagConfig, chunk_documents, format_sources, get_vectorstore


from openai import OpenAI



load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


st.set_page_config(page_title="DocChat Pro (SingleStore)", page_icon="🧠", layout="wide")


# ---------------- Helpers ----------------
def drop_table(table_name: str):
    conn = s2.connect(
        host=os.getenv("SINGLESTORE_HOST"),
        port=int(os.getenv("SINGLESTORE_PORT", "3306")),
        user=os.getenv("SINGLESTORE_USER"),
        password=os.getenv("SINGLESTORE_PASSWORD"),
        database=os.getenv("SINGLESTORE_DATABASE"),
    )
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS `{table_name}`;")
    conn.commit()
    conn.close()


# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Settings")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
)

model_name = st.sidebar.selectbox(
    "Model",
    ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
    index=0,
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

chunk_size = st.sidebar.slider("Chunk size", 300, 2000, 1200, 50)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 500, 200, 10)
top_k = st.sidebar.slider("Top-K retrieval", 2, 12, 8, 1)

table_name = st.sidebar.text_input(
    "SingleStore table name",
    value=os.getenv("SINGLESTORE_TABLE", "docchat_vectors"),
)

st.sidebar.divider()

uploaded = st.sidebar.file_uploader(
    "Upload docs (TXT/MD/PDF)",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True,
)

col1, col2 = st.sidebar.columns(2)
build_index = col1.button("🔁 Build / Upsert", use_container_width=True, disabled=not uploaded)
clear_chat = col2.button("🧹 Clear chat", use_container_width=True)

reset_kb = st.sidebar.button("🗑️ Reset Knowledge Base (DROP TABLE)", use_container_width=True)

if clear_chat:
    st.session_state.pop("messages", None)

if reset_kb:
    drop_table(table_name)
    st.session_state.pop("vectorstore", None)
    st.success(f"Dropped table `{table_name}` ✅ Now re-upload and Build/Upsert.")
    st.stop()


# ---------------- Main UI ----------------
st.title("🧠 DocChat Pro — SingleStore Vector DB")
st.caption("Upload docs → persist embeddings in SingleStore → chat with citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for title, snippet in m["sources"]:
                    st.markdown(f"**{title}**")
                    st.write(snippet)

# Build/Upsert
if build_index:
    if not api_key:
        st.error("Add your OpenAI API key in the sidebar.")
        st.stop()

    cfg = RagConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        table_name=table_name,
    )

    with st.spinner("Reading uploaded files…"):
        docs = uploaded_files_to_documents(uploaded)

    with st.spinner("Chunking documents…"):
        chunks = chunk_documents(docs, cfg)

    with st.spinner("Connecting to SingleStore…"):
        vs = get_vectorstore(table_name=table_name)

    st.info(f"Upserting {len(chunks)} chunks (large PDFs can take time)…")
    progress = st.progress(0)

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        vs.add_documents(chunks[i : i + batch_size])
        progress.progress(min((i + batch_size) / len(chunks), 1.0))

    st.session_state["vectorstore"] = vs
    st.success(f"Upsert complete ✅ Files: {len(docs)} | Chunks: {len(chunks)} | Table: {table_name}")


# Chat
user_q = st.chat_input("Ask a question about your uploaded documents…")

if user_q:
    if "vectorstore" not in st.session_state:
        st.warning("Upload documents and click **Build / Upsert** first.")
        st.stop()

    if not api_key:
        st.error("Add your OpenAI API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    llm = ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        streaming=True,
    )

    vs = st.session_state["vectorstore"]

    # Retriever
    retriever = vs.as_retriever(search_kwargs={"k": top_k})

    # Retrieve docs
    docs = retriever.get_relevant_documents(user_q)

    # 🔍 Debug retrieved chunks
    with st.expander("🔍 Retrieved chunks (debug)"):
        st.write(f"Retrieved: {len(docs)} chunks")
        for i, d in enumerate(docs, 1):
            st.markdown(f"### Chunk {i} — {d.metadata.get('source', '')}")
            st.write(d.page_content[:1500])

    context = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""

        messages = prompt.format_messages(context=context, question=user_q)

        for chunk in llm.stream(messages):
            token = getattr(chunk, "content", "") or ""
            answer += token
            placeholder.markdown(answer)

        sources = format_sources(docs)
        if sources:
            with st.expander("Sources"):
                for title, snippet in sources:
                    st.markdown(f"**{title}**")
                    st.write(snippet)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})