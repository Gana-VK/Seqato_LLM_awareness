import streamlit as st
import requests

st.set_page_config(page_title="Local LLM Chat", layout="wide")

st.title("🧠 Local LLM Chat with Mistral (Ollama)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

prompt = st.text_input("You:", "")

if st.button("Send") and prompt:
    st.session_state.chat_history.append(("You", prompt))
    
    response = requests.post("http://localhost:8000/chat", json={"prompt": prompt})
    reply = response.json().get("response", "No response")

    st.session_state.chat_history.append(("AI", reply))

for sender, message in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {message}")

