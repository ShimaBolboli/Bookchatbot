import streamlit as st
import os
import pandas as pd
import numpy as np
import asyncio
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from langchain.langchain import LangChain, Pipe
from langchain.helpers import wrap_transformer, PineconeEmbedding

# Initialize Pinecone
os.environ["PINECONE_API_KEY"] = '915e4017-1821-4418-b008-cff4b15b7677'
os.environ["PINECONE_ENVIRONMENT"] = 'us-east-1'
api_key = os.getenv("PINECONE_API_KEY")
valid_environment = os.getenv("PINECONE_ENVIRONMENT")
pc = Pinecone(api_key=api_key, environment=valid_environment)

# Set index details
index_name = 'books'
dimension = 384

# Check if index already exists
indexes = pc.list_indexes()
if index_name not in indexes.names():
    # Create the index if it doesn't exist
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='cosine',  # Adjust metric as needed
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Connect to the index
index = pc.Index(index_name)

# Load your dataset
df = pd.read_csv('books.csv')

# Clean and preprocess the dataset
df = df.dropna(subset=['title', 'authors'])  # Drop rows where 'title' or 'authors' are NaN
df['title'] = df['title'].astype(str)
df['authors'] = df['authors'].astype(str)
df['bookID'] = df['bookID'].astype(str)  # Ensure bookID is a string

# Initialize Hugging Face model for sentence embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Example model, replace with appropriate Hugging Face model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Initialize LangChain
lc = LangChain()

# Define pipeline for embedding and upserting
@lc.Pipe
def embed_and_upsert(data):
    embeddings = []
    for item in data:
        title_embedding = tokenizer(item['title'], return_tensors="pt")['input_ids']
        with torch.no_grad():
            title_embedding = model(title_embedding).last_hidden_state.mean(dim=1)
        embeddings.append(title_embedding.tolist())
    vectors = np.array(embeddings).astype(np.float32)
    ids = [item['bookID'] for item in data]
    meta = [{'title': item['title'], 'authors': item['authors']} for item in data]
    return {'ids': ids, 'vectors': vectors, 'meta': meta}

# Function to fetch top-k entries from Pinecone using a vector
async def fetch_top_k_entries(query_vector, k=3):
    try:
        # Query Pinecone index
        result = index.query(queries=[query_vector], top_k=k, include_metadata=True)

        top_k_entries = []
        for match in result['matches']:
            title = match['metadata']['title']
            authors = match['metadata']['authors']
            top_k_entries.append({
                'title': title,
                'authors': authors
            })

        return top_k_entries

    except Exception as e:
        print(f"Pinecone API Exception: {e}")
        return []

# Async wrapper function
async def process_query(query):
    # Embed the query using Hugging Face model
    query_embedding = tokenizer(query, return_tensors="pt")['input_ids']
    with torch.no_grad():
        query_embedding = model(query_embedding).last_hidden_state.mean(dim=1).numpy()

    # Fetch top-k entries
    results = await fetch_top_k_entries(query_embedding)
    return results

# Streamlit UI
st.title('Book Recommendation Chatbot')

query = st.text_input('Enter your query:', '')

if st.button('Search'):
    if query:
        st.write(f'Query Entered: {query}')
        
        # Display a placeholder for results
        result_placeholder = st.empty()
        result_placeholder.text("Searching for recommendations...")

        # Run async wrapper function using asyncio.run()
        results = asyncio.run(process_query(query))

        if results:
            result_placeholder.empty()
            st.write("Top Recommendations:")
            for result in results:
                st.write(f"- Title: {result['title']}, Authors: {result['authors']}")
        else:
            result_placeholder.empty()
            st.write("Sorry, no relevant recommendations found for your query.")
