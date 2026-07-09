from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore


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



extracted_data=load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks=text_split(filter_data)

embeddings = download_hugging_face_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



index_name = "custom-chatbot"  # change if desired

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)