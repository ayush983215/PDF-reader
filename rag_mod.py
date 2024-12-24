import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

def extract_pdf_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text


def chunk_text(text, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks


def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Example model
    embeddings = model.encode(chunks)
    return embeddings



def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def query_index(query, model, index, chunks, top_k=3):
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [chunks[i] for i in indices[0]]



def generate_response(prompt, api_key):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

class RAGModel:
    def __init__(self, pdf_path, embedding_model='all-MiniLM-L6-v2', api_key=None):
        self.api_key = api_key
        self.text = extract_pdf_text(pdf_path)
        self.chunks = chunk_text(self.text)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embeddings = embed_chunks(self.chunks)
        self.index = create_faiss_index(np.array(self.embeddings))

    def query(self, user_query, top_k=3):
        relevant_chunks = query_index(user_query, self.embedding_model, self.index, self.chunks, top_k)
        prompt = f"Using the following information: {relevant_chunks}, answer: {user_query}"
        return generate_response(prompt, self.api_key)

# Usage
pdf_path = '/Users/harshpundhir/Downloads/python_dev_resume.pdf'
api_key = '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'

rag_model = RAGModel(pdf_path, api_key=api_key)
response = rag_model.query("What is the summary of this PDF?")
print(response)
