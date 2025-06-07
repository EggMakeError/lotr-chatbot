import streamlit as st
import random
import re
from rag_bot import (
    load_documents,
    extract_characters,
    create_character_chain_with_memory
)

def set_background_and_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cinzel&display=swap');

        .stApp {
            background-image: url("https://wallpapercat.com/w/full/1/7/a/137052-3840x2160-desktop-4k-the-lord-of-the-rings-wallpaper.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: #d7c4a3;
            font-family: 'Cinzel', serif;
        }
        .css-1d391kg, .css-1kyxreq {
            background-color: rgba(0, 0, 0, 0.6) !important;
            border-radius: 8px;
            padding: 10px;
        }
        textarea[data-testid="stChatInputTextarea"] {
            background-color: #f5f1e7 !important;
            color: #4b3b2b !important;
            border: 2px solid #7a5c3e !important;
            border-radius: 8px !important;
            font-family: 'Cinzel', serif !important;
        }
        button[data-testid="stChatSendButton"] {
            background-color: #7a5c3e !important;
            color: #f5f1e7 !important;
            border: none !important;
            padding: 10px 20px !important;
            border-radius: 8px !important;
            font-family: 'Cinzel', serif !important;
            cursor: pointer !important;
            transition: background-color 0.3s ease !important;
        }
        button[data-testid="stChatSendButton"]:hover {
            background-color: #a38156 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_title="Fellowship of the Ring RAG Chatbot", layout="wide")
st.title("🧙 Welcome to a World of Wonder: Conversations in Middle-earth")
set_background_and_style()

if "docs_loaded" not in st.session_state:
    with st.spinner("Gathering tales from Middle-earth..."):
        documents = load_documents(["lotr-characters.pdf"])
        st.session_state.all_docs = documents
        st.session_state.characters = extract_characters(documents)
        st.session_state.docs_loaded = True

fellowship_names = list(st.session_state.characters.keys())

if "current_character" not in st.session_state:
    st.session_state.current_character = random.choice(fellowship_names)

if "qa_chain" not in st.session_state or st.session_state.qa_chain is None:
    try:
        st.session_state.qa_chain = create_character_chain_with_memory(
            st.session_state.current_character,
            st.session_state.all_docs,
            st.session_state.characters
        )
    except Exception as e:
        st.error(f"Error initializing character chain: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = {}

if st.session_state.current_character not in st.session_state.messages:
    greeting = st.session_state.characters[st.session_state.current_character]['greeting']
    st.session_state.messages[st.session_state.current_character] = [{
        "role": "assistant",
        "content": f"You find yourself before **{st.session_state.current_character}**.\n\n> {greeting}"
    }]

for message in st.session_state.messages.get(st.session_state.current_character, []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def detect_identity_claim(text):
    for name in fellowship_names:
        if re.search(rf"\b(I am|I'm|call me|my name is)\s+{name}\b", text, re.IGNORECASE):
            return name
    return None

def detect_character_switch(text):
    for name in fellowship_names:
        if re.search(rf"\b(speak to|talk to|switch to|see|bring me to)\s+{name}\b", text, re.IGNORECASE):
            return name
    return None

def detect_out_of_character_request(text):
    ooc_patterns = [
        r"break character", r"ignore previous instructions", r"just be yourself",
        r"you are not really", r"drop the act", r"speak as (?:ChatGPT|an AI|a bot)"
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in ooc_patterns)

def detect_past_conversation_with_character(text):
    for name in fellowship_names:
        if re.search(rf"\b(what did i talk about with|what did we talk about with|tell me about my conversation with)\s+{name}\b", text, re.IGNORECASE):
            return name
    return None

query = st.chat_input("Ask your question in the tongue of Middle-earth...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.setdefault(st.session_state.current_character, []).append({
        "role": "user",
        "content": query
    })

    new_char = detect_character_switch(query)
    response = None

    if new_char and new_char != st.session_state.current_character:
        st.session_state.current_character = new_char
        try:
            st.session_state.qa_chain = create_character_chain_with_memory(
                new_char,
                st.session_state.all_docs,
                st.session_state.characters
            )
        except Exception as e:
            st.error(f"Error switching character chain: {e}")
            st.session_state.qa_chain = None

        if new_char not in st.session_state.messages:
            greeting = st.session_state.characters[new_char]["greeting"]
            st.session_state.messages[new_char] = [{
                "role": "assistant",
                "content": f"You find yourself before **{new_char}**.\n\n> {greeting}"
            }]
        else:
            st.session_state.messages[new_char] = [
                m for m in st.session_state.messages[new_char]
                if not m["content"].startswith("You are now speaking to")
            ]

        switch_msg = {
            "role": "assistant",
            "content": f"You are now speaking to **{new_char}**."
        }
        st.session_state.messages[new_char].append(switch_msg)

        with st.chat_message("assistant"):
            st.markdown(switch_msg["content"])

    else:
        if detect_identity_claim(query):
            response = f"{st.session_state.current_character} narrows their eyes... \"I have traveled with him for a long time and know him better than most know themselves. Do not mock him so.\""
        elif detect_out_of_character_request(query):
            response = f"{st.session_state.current_character} looks sternly: \"I am not one for riddles of strange tongues. Speak plainly, or not at all.\""
        else:
            past_char = detect_past_conversation_with_character(query)
            if past_char and past_char != st.session_state.current_character:
                response = f"I don't know what you talked about with {past_char}."
            else:
                if st.session_state.get("qa_chain") is None:
                    response = "The character chain is still loading or not initialized."
                else:
                    with st.spinner(f"{st.session_state.current_character} is pondering..."):
                        response = st.session_state.qa_chain.run({"question": query})

        if response:
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages[st.session_state.current_character].append({
                "role": "assistant",
                "content": response
            })
