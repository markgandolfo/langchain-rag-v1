import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import SpacyTextSplitter
from sentence_transformers import SentenceTransformer, util
from langchain_community.llms import Ollama

os.system('clear')

ollama = Ollama(model="mistral")

YELLOW = "\033[1;33m"
END = "\033[0m"


# Step 0: Load the Model
# Load the Ollama model
ollama = Ollama(model="mistral")

print(f"{YELLOW}Model loaded: {ollama.model}{END}")

# Step 1: Load and Preprocess Text Files
# Load text files from a directory

loader = DirectoryLoader(path="./docs")
documents = loader.load()

print(f"{YELLOW}Number of documents loaded: {len(documents)}{END}")

question = "what's in my bathtub?"

# Split the documents into smaller chunks using SpacyTextSplitter
text_splitter = SpacyTextSplitter(chunk_size=1000)
chunks = text_splitter.split_documents(documents)

print(f"{YELLOW}Number of chunks created: {len(chunks)}{END}")

# Step 2: Create Document Embeddings
# Load the sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

for chunk in chunks:
    print(f"{YELLOW}chunk: {chunk.page_content}{END}")
    a = util.cos_sim(embedding_model.encode(question), embedding_model.encode(chunk.page_content))
    print(f"{YELLOW}what does this give me? {a}{END} ")
    print(chunk.page_content)


# Create embeddings for each chunk
embeddings = [embedding_model.encode(chunk.page_content) for chunk in chunks]

for embedding in embeddings:
    print(f"{YELLOW}embedding: {embedding.shape[0]}{END}")

# Step 3: Store Embeddings in a Vector Store
# Create a FAISS index
dimension = embeddings[0].shape[0]
print(f"{YELLOW}dimensions: {dimension}{END}")


