import streamlit as st
from pypdf import PdfReader
import io
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
import json
import re 
import httpx 
import asyncio

# --- Helper Functions ---

def sanitize_collection_name(name: str) -> str:
    """
    Sanitizes a string to be a valid ChromaDB collection name.
    - Replaces spaces and invalid characters with underscores.
    - Ensures the name is between 3 and 63 characters.
    - Ensures the name starts and ends with an alphanumeric character.
    """
    # Replace invalid characters with underscores
    sanitized_name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    
    # Ensure the name is not too long or too short
    if len(sanitized_name) > 63:
        sanitized_name = sanitized_name[:63]
    if len(sanitized_name) < 3:
        sanitized_name = sanitized_name.ljust(3, 'x')
    if not sanitized_name[0].isalnum():
        sanitized_name = 'a' + sanitized_name[1:]
    if not sanitized_name[-1].isalnum():
        sanitized_name = sanitized_name[:-1] + 'a'
        
    return sanitized_name


def get_pdf_text(pdf_bytes):
    """Extracts text from a PDF file provided as bytes."""
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Include page number in the text for citation
                text += f"--- Page {i + 1} ---\n{page_text}\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def get_text_chunks(text):
    """Splits a long text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_embeddings_model():
    """Loads the sentence transformer model."""
    return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")

def create_vector_store(text_chunks, collection_name):
    """Creates a Chroma vector store from text chunks."""
    embeddings = get_embeddings_model()
    vector_store = Chroma.from_texts(
        texts=text_chunks, 
        embedding=embeddings,
        collection_name=collection_name
    )
    return vector_store

async def get_comparison_from_gemini(query, context_A, context_B):
    """
    Generates a comparative analysis using the Gemini API.
    """
    prompt = f"""
    You are a specialized AI assistant for comparative analysis.
    Your task is to compare and contrast the information provided from two different sources based on a user's query.

    User Query: "{query}"

    ---
    Context from Corpus A:
    {context_A}
    ---
    Context from Corpus B:
    {context_B}
    ---

    Based on the provided contexts and the user query, generate a structured comparison.
    The output should be a JSON object with the following keys:
    - "similarities": A list of key similarities between the two corpora regarding the query.
    - "differences": A list of key differences.
    - "side_by_side_metrics": A list of dictionaries, where each dictionary represents a metric with keys "metric_name", "corpus_a_value", and "corpus_b_value". If no numeric metrics are found, this should be an empty list.
    - "summary": A brief overall summary of the comparison.

    Provide citations by referring to page numbers mentioned in the context (e.g., [Page 5]).
    """
    
    chatHistory = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {"contents": chatHistory}
    
    try:
        apiKey = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("GEMINI_API_KEY not found in secrets.toml. Please add it.")
        return None

    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(apiUrl, json=payload, headers={'Content-Type': 'application/json'}, timeout=120)
            response.raise_for_status()
            result = response.json()
        
        if result.get('candidates'):
            raw_json = result['candidates'][0]['content']['parts'][0]['text']
            cleaned_json = raw_json.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_json)
        else:
            st.error("Error from Gemini API: No candidates found.")
            st.json(result)
            return None
    except httpx.HTTPStatusError as e:
        st.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Failed to process Gemini API response: {e}")
        return None

# --- Streamlit App UI ---
st.set_page_config(page_title="Comparison RAG Dashboard", layout="wide")
st.title("📊 Comparison RAG Dashboard")
st.write("Upload two documents to compare their content side-by-side.")

# Initialize session state for query count
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

# --- Sidebar for Document Uploads ---
with st.sidebar:
    st.header("Upload Documents")
    
    st.subheader("Corpus A")
    uploaded_file_A = st.file_uploader("Upload document for Corpus A", type="pdf", key="uploader_A")
    
    st.subheader("Corpus B")
    uploaded_file_B = st.file_uploader("Upload document for Corpus B", type="pdf", key="uploader_B")

    # Processing logic for Corpus A
    if uploaded_file_A:
        file_key_A = f"A-{uploaded_file_A.name}-{uploaded_file_A.size}"
        sanitized_collection_name_A = sanitize_collection_name(file_key_A)
        if st.session_state.get("processed_key_A") != file_key_A:
            with st.spinner('Processing Corpus A...'):
                raw_text_A = get_pdf_text(uploaded_file_A.getvalue())
                if raw_text_A:
                    chunks_A = get_text_chunks(raw_text_A)
                    st.session_state.vector_store_A = create_vector_store(chunks_A, collection_name=sanitized_collection_name_A)
                    st.session_state.processed_key_A = file_key_A
                    st.success("Corpus A is ready!")

    # Processing logic for Corpus B
    if uploaded_file_B:
        file_key_B = f"B-{uploaded_file_B.name}-{uploaded_file_B.size}"
        sanitized_collection_name_B = sanitize_collection_name(file_key_B)
        if st.session_state.get("processed_key_B") != file_key_B:
            with st.spinner('Processing Corpus B...'):
                raw_text_B = get_pdf_text(uploaded_file_B.getvalue())
                if raw_text_B:
                    chunks_B = get_text_chunks(raw_text_B)
                    st.session_state.vector_store_B = create_vector_store(chunks_B, collection_name=sanitized_collection_name_B)
                    st.session_state.processed_key_B = file_key_B
                    st.success("Corpus B is ready!")

# --- Main Content Area ---
if "vector_store_A" in st.session_state and "vector_store_B" in st.session_state:
    st.header("Ask a Comparative Question")
    user_query = st.text_input("e.g., 'Compare the revenue growth between the two companies in Q1'")

    if user_query:
        st.session_state.query_count += 1
        async def perform_comparison():
            with st.spinner("Retrieving and comparing..."):
                retriever_A = st.session_state.vector_store_A.as_retriever(search_kwargs={"k": 3})
                docs_A = retriever_A.get_relevant_documents(user_query)
                context_A = "\n\n".join([doc.page_content for doc in docs_A])

                retriever_B = st.session_state.vector_store_B.as_retriever(search_kwargs={"k": 3})
                docs_B = retriever_B.get_relevant_documents(user_query)
                context_B = "\n\n".join([doc.page_content for doc in docs_B])
                
                comparison_result = await get_comparison_from_gemini(user_query, context_A, context_B)

                if comparison_result:
                    st.subheader("Comparative Analysis")
                    st.markdown(f"**Summary:** {comparison_result.get('summary', 'N/A')}")
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Similarities")
                        for item in comparison_result.get('similarities', []):
                            st.markdown(f"- {item}")
                    
                    with col2:
                        st.markdown("#### Differences")
                        for item in comparison_result.get('differences', []):
                            st.markdown(f"- {item}")

                    st.markdown("---")
                    
                    metrics = comparison_result.get('side_by_side_metrics', [])
                    if metrics:
                        st.markdown("#### Side-by-Side Metrics")
                        try:
                            df = pd.DataFrame(metrics)
                            df['corpus_a_value'] = pd.to_numeric(df['corpus_a_value'], errors='coerce')
                            df['corpus_b_value'] = pd.to_numeric(df['corpus_b_value'], errors='coerce')
                            
                            st.table(df)

                            st.markdown("#### Metrics Visualization")
                            df_chart = df.rename(columns={
                                'corpus_a_value': uploaded_file_A.name,
                                'corpus_b_value': uploaded_file_B.name
                            })
                            st.bar_chart(df_chart.set_index('metric_name'))

                        except Exception as e:
                            st.warning("Could not display metrics table or chart.")
                            st.json(metrics)
                    
                    # **NEW**: Add user feedback widget
                    st.markdown("---")
                    feedback = st.feedback(
                        "thumbs",
                        key=f"feedback_{st.session_state.query_count}",
                    )
                    if feedback:
                        # In a real app, you would log this to a database or file
                        st.toast(f"Thank you for your feedback! You rated this '{feedback['score']}'.")


                    with st.expander("View Retrieved Context"):
                        st.markdown("#### Context from Corpus A")
                        st.text(context_A)
                        st.markdown("---")
                        st.markdown("#### Context from Corpus B")
                        st.text(context_B)
        
        asyncio.run(perform_comparison())

else:
    st.info("Please upload a document for both Corpus A and Corpus B to begin.")
