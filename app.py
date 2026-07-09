from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import src.prompt as prompt_module
import os

system_prompt = getattr(
    prompt_module,
    "system_prompt",
    getattr(prompt_module, "SYSTEM_PROMPT", "You are a helpful assistant.")
)


app = Flask(__name__)


load_dotenv()


def get_env_var(name: str) -> str:
    value = os.environ.get(name, '') or ''
    value = value.strip()
    if value.startswith(('"', "'")) and value.endswith(('"', "'")):
        value = value[1:-1]
    return value.replace('\n', '').replace('\r', '').strip()


PINECONE_API_KEY = get_env_var('PINECONE_API_KEY')
OPENAI_API_KEY = get_env_var('OPENAI_API_KEY')

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    missing = [name for name, value in (
        ('PINECONE_API_KEY', PINECONE_API_KEY),
        ('OPENAI_API_KEY', OPENAI_API_KEY),
    ) if not value]
    raise EnvironmentError(
        f"Missing environment variables: {', '.join(missing)}. "
        "Add them to .env or export them before running."
    )

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embeddings = download_hugging_face_embeddings()


index_name = "custom-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)



retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


chatModel = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{context}\n\nQuestion: {input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
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
    app.run(host="0.0.0.0", port= 8080, debug= True)