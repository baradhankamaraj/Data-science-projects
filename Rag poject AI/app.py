from augmentation import self_rag
import logging

logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_core").setLevel(logging.ERROR)
from langchain_core._api.deprecation import LangChainDeprecationWarning
import warnings

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
import streamlit as st


st.title("Government Scheme Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
question = st.chat_input("Ask about Government Schemes...")

if question:

    # Display user message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Display assistant response
    with st.chat_message("assistant"):

        with st.status("Processing...", expanded=True) as status:

            status.write("🔍 Retrieving documents...")

            result = self_rag(question)

            status.write("🤖 Generating answer...")

            status.write("📊 Evaluating answer...")

            status.update(
                label="Completed",
                state="complete"
            )

        if isinstance(result, dict):
            st.markdown(result.get("Answer", "No answer returned."))

            if result.get("Evaluation"):
                with st.expander("Evaluation"):
                    st.json(result["Evaluation"])
        else:
            st.error(str(result))

        # if "Evaluation" in result:
        #     with st.expander("Evaluation"):
        #         st.json(result["Evaluation"])

    # Save assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["Answer"]
        }
    )