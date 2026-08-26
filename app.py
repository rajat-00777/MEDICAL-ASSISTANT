from flask import Flask, render_template, jsonify, request, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from src.prompt import system_prompt, contextualize_q_system_prompt
import os
import uuid


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-only-fallback-change-me')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

app.config['SECRET_KEY'] = SECRET_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medibot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4}
)

chatModel = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.4,
)

# --- History-aware retriever: rewrites follow-up questions using chat
# history into standalone questions before hitting Pinecone. ---
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    chatModel, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chatModel, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# --- In-memory chat history store, keyed by session_id. ---
# NOTE: this is server-memory only — history is lost on restart, and won't
# scale across multiple server workers/processes. Fine for local dev/small
# deployments. For production, swap this dict for Redis-backed storage.
chat_store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_store:
        chat_store[session_id] = InMemoryChatMessageHistory()
    return chat_store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


STARTER_QUESTIONS = [
    "What is acne and how is it treated?",
    "What are the symptoms of a fungal infection?",
    "What causes seasonal allergies?",
    "How is a fever normally treated?",
]


def format_sources(context_docs):
    """Turn retrieved Document objects into a deduped list of source citations."""
    seen = set()
    sources = []
    for doc in context_docs:
        filename = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page")
        key = (filename, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"file": filename, "page": page})
    return sources


@app.route("/")
def index():
    # Assign a stable per-browser session id (used as the chat-history key).
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template('chat.html', starter_questions=STARTER_QUESTIONS)


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    session_id = session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id

    print(f"[{session_id}] User:", msg)

    response = conversational_rag_chain.invoke(
        {"input": msg},
        config={"configurable": {"session_id": session_id}},
    )
    answer = response["answer"]
    sources = format_sources(response.get("context", []))

    print(f"[{session_id}] Bot:", answer)

    return jsonify({
        "answer": answer,
        "sources": sources
    })


@app.route("/new_chat", methods=["POST"])
def new_chat():
    """Clears server-side memory for the current session and starts fresh."""
    session_id = session.get("session_id")
    if session_id and session_id in chat_store:
        del chat_store[session_id]
    session["session_id"] = str(uuid.uuid4())
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)