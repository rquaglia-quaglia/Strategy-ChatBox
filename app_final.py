import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Strategy Chatbot", layout="wide")
st.title("🎓 MBA Strategy Chatbot – Value Proposition Module")

# Load GPT model and FAISS vectorstore
@st.cache_resource
def load_qa_final():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain

# Input field
query = st.text_input("Ask me a question about strategy, value propositions, or Lumora case:")

if query:
    qa = load_qa_final()
    with st.spinner("Thinking..."):
        result = qa(query)
        st.markdown("### 🤖 Answer")
        st.write(result["result"])
