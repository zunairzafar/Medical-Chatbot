from dotenv import load_dotenv
import os 
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
pinecone_api_key = os.getenv("PINECONE_API_KEY")


extracted_data = load_pdf_file("data/")
filter_data = filter_to_minimal_docs(extracted_data)
chunked_data = text_split(filter_data)
from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.embed_documents([doc.page_content for doc in chunked_data])


if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is missing. Add it to your .env file.")

index_name = os.getenv("medical-chatbot", "medical-chatbot")
#A check to see if index exists, if not it creates one
pc = Pinecone(api_key=pinecone_api_key)
existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=len(embeddings[0]),
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        ),
    )

index = pc.Index(index_name)
docsearch = PineconeVectorStore.from_documents(
    documents = chunked_data,
    embedding=embedding_model,
    index_name=index_name
)

