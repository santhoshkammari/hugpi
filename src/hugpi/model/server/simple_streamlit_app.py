import streamlit as st
import requests
from typing import Generator

def get_api_response(prompt, stream=True, conversation=False, url="http://0.0.0.0:11435/v1/generate",
                     web_search = False) -> Generator:
    data = {
        "prompt": prompt,
        "stream": stream,
        "conversation": conversation,
        "websearch":web_search
    }
    response = requests.post(url, json=data, stream=stream)
    for chunk in response.iter_content(decode_unicode=True):
        if chunk:
            yield chunk

def streamlit_app():
    st.title("Chat with Llama-3.1-Nemotron-70B")

    # Add a separator
    st.markdown("---")

    # Chat messages container
    chat_container = st.container()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if 'conversation_mode' not in st.session_state:
        st.session_state.conversation_mode = True
    if 'web_search' not in st.session_state:
        st.session_state.web_search = False

    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What is your message?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in get_api_response(prompt, conversation=st.session_state.conversation_mode,
                                              web_search = st.session_state.web_search):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
        # Create a container for toggles and clear button
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.conversation_mode = st.toggle("Conversation Mode", value=True)
        with col2:
            st.session_state.web_search = st.toggle("Enable Web Search", value=False)

if __name__ == "__main__":
    streamlit_app()