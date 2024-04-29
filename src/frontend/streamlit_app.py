import requests

import streamlit as st


def query_rag(question: str, k: int = 5):
    response = requests.get(
        "http://localhost:5656/scripture_references",
        params={"question": question, "k": k},
    )
    return response.json()

def markdownify_list(lst):
    return "\n".join([f"- **{item.split(':', 2)[0]}:{item.split(':', 2)[1]}**: {item.split(':', 2)[2]}" for item in lst])

st.title("Scripture Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question to search the scriptures."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = query_rag(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(markdownify_list(response))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})