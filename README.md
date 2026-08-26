# 🩺 Medical Chatbot

An AI-powered Medical Chatbot built using **LangChain**, **Groq**, **Pinecone**, and **Flask**. The chatbot answers medical-related questions by retrieving information from a medical knowledge base using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- 📄 PDF-based medical knowledge
- 🤖 AI-powered question answering
- 🔍 Retrieval-Augmented Generation (RAG)
- ⚡ Fast response generation with follow up question
- 🎨 Simple and responsive web interface
- 🌐 Flask web application

---

## 🛠 Tech Stack

- Python
- Flask
- LangChain
- Pinecone
- GROQ AI
- HTML
- CSS

---

## 📁 Project Structure

```
MEDICAL-CHATBOT/
│
├── data/
├── research/
├── src/
│   ├── helper.py
│   ├── prompt.py
│   └── __init__.py
│
├── static/
├── templates/
├── app.py
├── setup.py
├── requirements.txt
├── store_index.py
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/uniic7777/MEDICAL-CHATBOT.git
cd MEDICAL-CHATBOT
```

### Create a virtual environment with python version 3.10.10

```bash
python -m venv .venv
```

Activate it

**Windows**

```bash
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

Example:

```env
PINECONE_API_KEY=your_api_key
GROQ_API_KEY=your_api_key
```

---

## ▶️ Run the Application



## ▶️ Setup the Vector Database
```bash
python store_index.py
```




```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:8080
```



## 📌 Future Improvements

- Chat history
- User authentication
- Voice interaction
- Multiple PDF support
- Better UI/UX





