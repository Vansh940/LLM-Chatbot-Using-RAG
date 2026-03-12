import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
from threading import Thread

st.set_page_config(page_title="LLM Chatbot", page_icon="🤖")

st.title("🤖 Local LLM Chatbot")

# -----------------------------
# Load model (cached)
# -----------------------------
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
        dtype=torch.float16
    )

    model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()

# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# User input
# -----------------------------
prompt = st.chat_input("Ask something...")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Prompt formatting
    input_prompt = f"<user>: {prompt}\n<assistant>:"

    inputs = tokenizer(input_prompt, return_tensors="pt").to(device)

    # Streaming setup
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=1000,
        max_length=None,
        do_sample=False,
        use_cache=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    # Start generation thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    with st.chat_message("assistant"):

        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("🤖 Thinking..."):

            for new_text in streamer:
                full_response += new_text
                message_placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )