import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import SpacyTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.llms import Ollama
#ollama = Ollama(model="phi3")
ollama = Ollama(model="mistral")

YELLOW = "\033[1;33m"
END = "\033[0m"

# Step 1: Load and Preprocess Text Files
# Load text files from a directory

loader = DirectoryLoader(path="./docs")
documents = loader.load()

print(f"{YELLOW}Number of documents loaded: {len(documents)}{END}")

# Split the documents into smaller chunks using SpacyTextSplitter
text_splitter = SpacyTextSplitter(chunk_size=1000)
chunks = text_splitter.split_documents(documents)

print(f"{YELLOW}Number of chunks created: {len(chunks)}{END}")

# Step 2: Create Document Embeddings
# Load the sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for each chunk
embeddings = [embedding_model.encode(chunk.page_content) for chunk in chunks]

# Step 3: Store Embeddings in a Vector Store
# Create a FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(np.array(embeddings))

# Step 4: Implement the Retrieval Mechanism
def retrieve_documents(query, k=5):
    # Create an embedding for the query
    query_embedding = embedding_model.encode(query).reshape(1, -1)

    # Search the index for the top-k nearest neighbors
    distances, indices = index.search(query_embedding, k)

    # Retrieve the corresponding documents
    retrieved_documents = [chunks[i].page_content for i in indices[0]]
    return retrieved_documents

# Step 5: Generate Answers Using the Language Model
def generate_answer(query, retrieved_docs):
    # Concatenate the retrieved documents
    context = " ".join(retrieved_docs)

    # Generate a response using the Ollama model
    response = ollama.generate(
        prompts=[f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"]
    )

    return response.generations[0][0].text

# Step 6: Create the Question Answering System
def answer_question(query):
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query)

    # Generate an answer using the language model
    answer = generate_answer(query, retrieved_docs)

    return answer

query = "How much money do i have?"
answer = answer_question(query)
print("\nAnswer:", answer)

