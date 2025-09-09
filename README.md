#### Dual Insight Engine: A Comparative RAG Dashboard
An advanced AI-powered dashboard that performs a side-by-side comparative analysis of two documents using a dual-corpus Retrieval-Augmented Generation (RAG) architecture.

## 🚀 Overview
Standard Large Language Models (LLMs) can answer questions based on their training data, but they struggle to perform direct, grounded comparisons between specific, user-provided documents. Dual Insight Engine solves this problem by creating a specialized analytical tool.

Instead of just asking a question to a generic model, you can upload two distinct PDF documents (e.g., a Tesla vs. a Ford quarterly report, two different research papers, or competing policy drafts). The application then retrieves the most relevant information from both sources simultaneously and uses a powerful LLM to generate a structured, side-by-side comparison, complete with metrics, summaries, and data visualizations.

## Tech Stack ##
Frontend: Streamlit
Core Language: Python
AI/LLM: Google Gemini API
Vector Database: Chroma
LLM Orchestration: LangChain
PDF Parsing: pypdf

## ✨ Features
Dual Document Upload: A clean interface to upload two separate PDF documents (Corpus A and Corpus B).

Comparative RAG Pipeline: Retrieves relevant context from both documents in parallel to answer a user's query.

Structured JSON Output: Prompts the LLM to return a detailed analysis in a reliable JSON format, ensuring consistent and parsable results.

Side-by-Side Analysis: Clearly displays the similarities and differences between the two documents.

Data Visualization: Automatically generates bar charts to visually compare any numerical metrics found in the texts.

Source Citations: All generated insights are grounded in the provided documents, with page numbers included in the context to ensure traceability.

User Feedback: A simple "thumbs up/down" mechanism to evaluate the quality of each generated comparison.

## ⚙️ How It Works
The application follows a modern Retrieval-Augmented Generation workflow:

Upload & Process: The user uploads two PDF files. The system extracts the text from each and includes page numbers for citations.

Chunk & Embed: The text from each document is split into smaller, semantically meaningful chunks. A sentence-transformer model then converts these chunks into vector embeddings.

Index: The embeddings for each document are stored in two separate, independent collections within a ChromaDB vector store.

Dual Retrieval: When a user asks a comparative question, the system queries both vector stores to find the top-k most relevant chunks from each document.

Augment & Generate: The retrieved chunks from both documents are injected into a specialized prompt. This augmented prompt is sent to the Gemini API, instructing it to perform a comparative analysis and return a structured JSON object.

Display: The Streamlit frontend parses the JSON response to display the summary, similarities.



