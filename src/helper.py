from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def filter_to_minimal_docs(documents: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in documents:
        isrc = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": isrc}
            )
        )
    return minimal_docs

def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap=50)
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk