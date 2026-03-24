from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY



from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model
)



retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


chatModel = ChatGroq(
    model="llama-3.1-8b-instant",  # or mixtral-8x7b-32768
    api_key=os.getenv("GROQ_API_KEY")
)
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an Medical Assistant for question-answering tasks."
    "Use the following pieces of retrieved context to answer."
    "The question: If you don't know the answer, say you don't know."
    "Use three sentences maximum to answer the question."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """API endpoint for chat interface - accepts JSON and returns JSON"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'response': 'Error: No message provided',
                'error': True
            }), 400
        
        user_message = data['message'].strip()
        
        if not user_message:
            return jsonify({
                'response': 'Error: Empty message',
                'error': True
            }), 400
        
        print(f"User Message: {user_message}")
        
        # Invoke the RAG chain
        response = rag_chain.invoke({"input": user_message})
        answer = response.get("answer", "No response generated")
        
        print(f"Bot Response: {answer}")
        
        return jsonify({
            'response': answer,
            'error': False
        })
    
    except Exception as e:
        print(f"Error in api_chat: {str(e)}")
        return jsonify({
            'response': f'Error: {str(e)}',
            'error': True
        }), 500


@app.route("/get", methods=["GET", "POST"])
def chat():
    """Legacy endpoint - kept for backward compatibility"""
    try:
        msg = request.form.get("msg", "")
        if not msg:
            return "Error: No message provided"
        
        print(f"User Message (legacy): {msg}")
        response = rag_chain.invoke({"input": msg})
        print(f"Response (legacy): {response['answer']}")
        return str(response["answer"])
    
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)