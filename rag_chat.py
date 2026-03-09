import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from threading import Thread

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 RAG Powered Chatbot")

# -------------------------
# Load Model
# -------------------------

@st.cache_resource
def load_model():

    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path = "./chatbot_model"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )

    model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return tokenizer, model, device


tokenizer, model, device = load_model()

# -------------------------
# Load Embeddings
# -------------------------

@st.cache_resource
def load_embeddings():

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    with open("knowledge/data.txt", "r", encoding="utf-8") as f:
        docs = f.read().split("\n\n")

    embeddings = embedder.encode(docs, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return embedder, docs, index


embedder, docs, index = load_embeddings()

# -------------------------
# Chat history
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.info("💬 Ask a question to start the conversation")

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------
# User input
# -------------------------

prompt = st.chat_input("Ask a question")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # -------------------------
    # RAG Retrieval
    # -------------------------

    query_vector = embedder.encode([prompt], convert_to_numpy=True)

    D, I = index.search(query_vector, k=3)

    context = "\n".join([docs[i] for i in I[0]])

    # -------------------------
    # Prompt
    # -------------------------

    input_prompt = f"""
Answer using ONLY the context.

Context:
{context}

Question:
{prompt}

Answer:
"""

    inputs = tokenizer(input_prompt, return_tensors="pt").to(device)

    # -------------------------
    # Streaming
    # -------------------------

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=300,
        mx_length=None,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # -------------------------
    # Assistant response
    # -------------------------

    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        message_placeholder.markdown("🤖 **Thinking...**")

        full_response = ""

        for new_text in streamer:
            full_response += new_text
            message_placeholder.markdown(full_response + "▌")

        # Final response with underline
        final_response = full_response + "\n\n---"

        message_placeholder.markdown(final_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": final_response}
    )