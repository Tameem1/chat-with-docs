__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import torch
import subprocess
import streamlit as st
#from run_localGPT import load_model
from langchain.vectorstores.chroma import Chroma
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_fireworks import ChatFireworks

def model_memory():
    # Adding history to the model.
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return prompt, memory


# Sidebar contents
with st.sidebar:
    st.title("روبوت المحادثة مع قانون")
    st.markdown(
        """
    روبوت دردشة ذكي يقدم أجوبة قانونية مستندة إلى قوانين المملكة الأردنية الهاشمية، يتيح للمستخدمين طرح الأسئلة والحصول على إجابات دقيقة وسريعة. 
     
 
    """
    )
    add_vertical_space(5)
    st.write("")


if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"


# if "result" not in st.session_state:
#     # Run the document ingestion process.
#     run_langest_commands = ["python", "ingest.py"]
#     run_langest_commands.append("--device_type")
#     run_langest_commands.append(DEVICE_TYPE)

#     result = subprocess.run(run_langest_commands, capture_output=True)
#     st.session_state.result = result

# Define the retreiver
# load the vectorstore
if "EMBEDDINGS" not in st.session_state:
    EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})
    st.session_state.EMBEDDINGS = EMBEDDINGS

if "DB" not in st.session_state:
    DB = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=st.session_state.EMBEDDINGS,
        client_settings=CHROMA_SETTINGS,
    )
    st.session_state.DB = DB

if "RETRIEVER" not in st.session_state:
    RETRIEVER = DB.as_retriever()
    st.session_state.RETRIEVER = RETRIEVER

if "LLM" not in st.session_state:
    LLM = ChatFireworks(
        api_key="fw_3ZfGXeDhjJfUxVHUVRBDfMeU",
        model="accounts/fireworks/models/qwen2p5-coder-32b-instruct",
        temperature=0.7,
        max_tokens=1500,
        top_p=1.0,
    )
    st.session_state["LLM"] = LLM


if "QA" not in st.session_state:
    prompt, memory = model_memory()

    QA = RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=RETRIEVER,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    st.session_state["QA"] = QA

st.title("برنامج الدردشة القانوني 💬")
# Create a text input box for the user
prompt = st.text_input("أدخل سؤالك هنا")
# while True:

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = st.session_state["QA"](prompt)
    answer, docs = response["result"], response["source_documents"]
    # ...and write it out to the screen
    st.write(answer)

    # # With a streamlit expander
    # with st.expander("Document Similarity Search"):
    #     # Find the relevant pages
    #     search = st.session_state.DB.similarity_search_with_score(prompt)
    #     # Write out the first
    #     for i, doc in enumerate(search):
    #         # print(doc)
    #         st.write(f"Source Document # {i+1} : {doc[0].metadata['source'].split('/')[-1]}")
    #         st.write(doc[0].page_content)
    #         st.write("--------------------------------")
