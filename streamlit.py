import streamlit as st
import llm

# Page Configs
st.set_page_config(page_title="PrivateGPT", page_icon="😛")
st.title("Hello World 😛")

# Session
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# File Uploader
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )


def save_message(role, message):
    st.session_state["messages"].append({"role": role, "message": message})


def send_message(role, message, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(role, message)


def show_messages():
    for message in st.session_state["messages"]:
        send_message(
            message["role"],
            message["message"],
            save=False,
        )


if file:
    file_path = llm.upload_file(file)
    retriever = llm.embed_file(file_path)
    send_message("ai", "Hello World !", save=False)
    show_messages()
    user_message = st.chat_input("send a message to ai")
    if user_message:
        send_message("human", user_message)
        ai_message = llm.execute_chain(retriever, user_message)
        send_message("ai", ai_message)
else:
    st.session_state["messages"] = []
