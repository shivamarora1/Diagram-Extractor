import streamlit as st
import botocore
from streamlit.logger import get_logger
import utils

logger = get_logger(__name__)


@st.cache_data(show_spinner=False)
def ask_doc_gpt(document_name, question):
    try:
        context = utils.get_relevant_context_from_collection(document_name, question)
        prompt = f"""{context}

Answer this question: {question}

- If it is required include the image url from context also.
- Answer to question should be brief unless user specifically ask for detailed answer.
- Final answer should be in Markdown text.
"""
        result = utils.prompt_llm(prompt)
        return result
    except botocore.exceptions.ClientError as error:
        logger.error("error in calling bedrock: ", str(error))
        return "Pls try again later. Error in AWS service"
    except Exception as e:
        logger.error("error in calling bedrock: ", str(e))
        return "Pls try again later."


SELECT_SUPPORTED_DOCS = [
    {
        "name": "tata_punch_owner_manual.pdf",
        "link": "https://b-public.s3.us-west-2.amazonaws.com/document-extractor/tata_punch_owner_manual/tata_punch_owner_manual.pdf",
    },
    {
        "name": "Raspberry_short_vers.pdf",
        "link": "https://drive.google.com/file/d/1jUB6AWlNztOToOLJcWi0W4fkZ4yGyYU2/view?usp=drive_link",
    },
]

## * hide sidebar close icon.
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)


KEY_SELECTED_DOCUMENT = "selected_document"

CHAT_ENABLE = False
if KEY_SELECTED_DOCUMENT not in st.session_state:
    st.session_state[KEY_SELECTED_DOCUMENT] = ""

if st.session_state[KEY_SELECTED_DOCUMENT]:
    CHAT_ENABLE = True


def select_document(name):
    st.session_state[KEY_SELECTED_DOCUMENT] = name


if CHAT_ENABLE:
    st.success(
        f"""The *{st.session_state[KEY_SELECTED_DOCUMENT]}* has been successfully selected.\
        You can now start asking questions about it.""",
        icon="✅",
    )
else:
    st.info("Select a document to start the chart.", icon="👈")


with st.sidebar:
    st.header("Available documents", divider="gray")

    for doc in SELECT_SUPPORTED_DOCS:
        with st.container():
            col1, col2, col3 = st.columns(
                [0.7, 0.15, 0.15], vertical_alignment="center"
            )
            name = doc["name"]
            with col1:
                st.write(f"{name}", unsafe_allow_html=True)
            with col2:
                st.button(
                    label="🔗",
                    help="select",
                    key=name,
                    on_click=select_document,
                    args=[doc["name"]],
                )
            with col3:
                st.button(
                    label="↗",
                    help="preview document",
                    key=f"{name}_preview",
                    on_click=utils.preview_document,
                    args=[doc["link"]],
                )


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you?", disabled=not CHAT_ENABLE):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking🌀..."):
            selected_doc = st.session_state[KEY_SELECTED_DOCUMENT]
            logger.info("selected document is %s | Prompt is %s", selected_doc, prompt)
            response = ask_doc_gpt(selected_doc, prompt)
            logger.info("received response: %s", response)
        st.write_stream(utils.streamed_response_generator(response))
    st.session_state.messages.append({"role": "assistant", "content": response})
