# Government Scheme Assistant (RAG)

## Overview

Government Scheme Assistant is a Retrieval-Augmented Generation (RAG) application that helps users query Indian government schemes using natural language. Instead of relying only on a large language model, the application retrieves relevant information from official scheme documents and uses it to generate accurate, context-aware answers.

The project is built using **LangChain**, **FAISS**, and **Groq Llama 3.3 70B** to provide fast and reliable responses.

---

## Features

* Ask questions about Indian government schemes in natural language.
* Retrieval-Augmented Generation (RAG) for more accurate responses.
* Semantic document search using FAISS.
* Context-aware answer generation using Groq Llama 3.3 70B.
* Automatic relevance filtering for retrieved documents.
* Answer critique based on:

  * Faithfulness
  * Relevance
  * Completeness
  * Conciseness
  * Safety

---

## Technologies Used

* Python
* LangChain
* FAISS
* Groq API
* Llama 3.3 70B Versatile
* HuggingFace Embeddings
* python-dotenv

---

## Project Structure

```
Government-Scheme-RAG/
│
├── data/                 # Government scheme documents
├── vectorstore/          # FAISS index
├── prompts/              # Prompt templates
├── app.py                # Main application
├── requirements.txt
├── .env.example
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd Government-Scheme-RAG
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```
GROQ_API_KEY=your_groq_api_key
```

---

## Running the Project

```bash
python app.py
```

---

## Example Questions

* What is PM-KISAN Scheme?
* Who is eligible for PM Fasal Bima Yojana?
* What are the benefits of the Soil Health Card Scheme?
* How can I apply for e-NAM?
* What documents are required for Kisan Credit Card?

---

## How It Works

1. Government scheme documents are loaded.
2. Documents are converted into embeddings.
3. Embeddings are stored in a FAISS vector database.
4. User questions are converted into embeddings.
5. The most relevant documents are retrieved.
6. The retrieved context is passed to the Llama 3.3 70B model.
7. The model generates a context-grounded response.
8. The response is evaluated using a critique step for quality.

---

## Future Improvements

* Web interface using Streamlit or Gradio
* Multi-language support
* Voice-based interaction
* Integration with live government scheme updates
* Citation of source documents
* User feedback mechanism

---

## Author

**Baradhan Kamaraj**

* Aspiring Data Scientist
* Skilled in Machine Learning, Deep Learning, NLP, and Retrieval-Augmented Generation (RAG)
