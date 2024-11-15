from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings, load_pdf_file, text_split
# Remove this outdated import: from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangChainPinecone  # Corrected import
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

# Environment variables setup
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Set environment variables (ensure they are set correctly in your .env file)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone index setup (use the existing index)
index_name = "medicalbot"
docsearch = LangChainPinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever and LLM chain
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create ChatGroq model instance and set up the prompt
llm = ChatGroq(temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create document chain and retrieval chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
