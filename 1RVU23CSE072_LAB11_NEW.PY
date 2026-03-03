import os
import streamlit as st
import numpy as np
import networkx as nx
from pyvis.network import Network

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings


# ================= PAGE CONFIG =================
st.set_page_config(page_title="Graph RAG System")
st.title("Graph RAG Demo")


# ================= SIDEBAR =================
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if api_key:
    os.environ["GROQ_API_KEY"] = api_key

# ================= SESSION STATE =================
if "graph" not in st.session_state:
    st.session_state.graph = nx.Graph()

if "node_embeddings" not in st.session_state:
    st.session_state.node_embeddings = {}

if "messages" not in st.session_state:
    st.session_state.messages = []


# ================= MODELS =================
def get_llm():
    return ChatGroq(
        model_name="openai/gpt-oss-120b",
        temperature=0
    )


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ================= TRIPLE EXTRACTION =================
def extract_triples(text, llm):
    prompt = f"""
Extract relationships as triples in this format:
Subject | Relationship | Object

Text:
{text}
    """
    response = llm.invoke(prompt).content

    triples = []
    for line in response.split("\n"):
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


# ================= GRAPH BUILD =================
def build_graph_from_text(text):

    llm = get_llm()
    embed_model = get_embedding_model()

    triples = extract_triples(text, llm)

    G = st.session_state.graph
    node_embeddings = st.session_state.node_embeddings

    for subj, rel, obj in triples:
        G.add_node(subj)
        G.add_node(obj)
        G.add_edge(subj, obj, relation=rel)

    # Embed new nodes
    for node in G.nodes():
        if node not in node_embeddings:
            node_embeddings[node] = embed_model.embed_query(node)

    return triples


# ================= GRAPH VISUAL =================
def render_graph(G):

    net = Network(height="500px", width="100%", notebook=False)
    net.from_nx(G)

    for edge in G.edges(data=True):
        net.edges[-1]["title"] = edge[2].get("relation", "")

    net.save_graph("graph.html")
    with open("graph.html", "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=500)


# ================= RETRIEVAL =================
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def graph_retrieve(query):

    embed_model = get_embedding_model()
    node_embeddings = st.session_state.node_embeddings
    G = st.session_state.graph

    query_vec = embed_model.embed_query(query)

    scores = []
    for node, vec in node_embeddings.items():
        sim = cosine_sim(query_vec, vec)
        scores.append((node, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in scores[:2]]

    context = []
    for node in top_nodes:
        for neighbor in G.neighbors(node):
            rel = G[node][neighbor]["relation"]
            context.append(f"{node} {rel} {neighbor}")

    return "\n".join(context)


# ================= QA =================
def answer_question(question):

    llm = get_llm()
    context = graph_retrieve(question)

    prompt = f"""
Answer the question using only the graph context below.

Graph Context:
{context}

Question:
{question}
    """

    return llm.invoke(prompt).content


# ================= TEXT INPUT =================
st.subheader("Add Knowledge to Graph")
text_input = st.text_area("Enter text to extract knowledge")

if st.button("Build Graph"):
    if not api_key:
        st.warning("Enter API key first")
    elif text_input:
        triples = build_graph_from_text(text_input)
        st.success("Graph updated")
        st.write("Extracted Triples:", triples)


# ================= GRAPH DISPLAY =================
if len(st.session_state.graph.nodes()) > 0:
    st.subheader("Graph Visualization")
    render_graph(st.session_state.graph)


# ================= CHAT =================
st.subheader("Ask Questions")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask about the graph"):
    if not api_key:
        st.warning("Enter API key first")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        answer = answer_question(question)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
