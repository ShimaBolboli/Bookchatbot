import streamlit as st
import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from tqdm import tqdm
import torch

# Initialize Pinecone
os.environ["PINECONE_API_KEY"] =  '********-****-****-****-************' 
os.environ["PINECONE_ENVIRONMENT"] = 'YOUR_PINECONE_ENVIRONMENT'
api_key = os.getenv("PINECONE_API_KEY")
valid_environment = os.getenv("PINECONE_ENVIRONMENT")

pc = Pinecone(api_key=api_key, environment=valid_environment)

# Set index details
index_name = 'books'
dimension = 384
text_key = 'title'
authors_key = 'authors'

# Check if index already exists
if index_name not in pc.list_indexes().names():
    # Create the index if it doesn't exist
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='cosine',  # Adjust metric as needed
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
 

# Connect to the index
index = pc.Index(index_name)

print ('sdsfsdfs',index)

# Load your dataset with proper handling of NaN values
df = pd.read_csv('books.csv')
print("Dataset shape:", df.shape)  # Log the shape of the dataset
print("Dataset head:\n", df.head())  # Log the first few rows for column inspection

# Clean and preprocess the dataset
df = df.dropna(subset=['title', 'authors'])  # Drop rows where 'title' or 'authors' are NaN
df['title'] = df['title'].astype(str)
df['authors'] = df['authors'].astype(str)
df['bookID'] = df['bookID'].astype(str)  # Ensure bookID is a string


# Initialize Hugging Face model for sentence embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Example model, replace with appropriate Hugging Face model
model_kwargs = {'device': 'cpu'}  # Set device to 'cuda' if GPU is available
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(model_name=model_name,
                                       model_kwargs=model_kwargs,
                                       encode_kwargs=encode_kwargs)

# Initialize Hugging Face model for text generation
generator = pipeline("text-generation", model="distilgpt2", tokenizer="distilgpt2", device=0 if model_kwargs['device'] == 'cuda' else -1)  # Example generation model, set device accordingly

# Initialize summarization pipeline
summarizer = pipeline(
    "summarization",
    model="pszemraj/long-t5-tglobal-base-16384-book-summary",
    device=0 if torch.cuda.is_available() else -1,
)

# Function to upsert data into Pinecone
def upsert_data():
    try:
        batch_size = 50
        vectors = []
        for start_idx in tqdm(range(0, len(df), batch_size), desc="Upserting batches"):
            end_idx = min(start_idx + batch_size, len(df))
            batch_texts = df[['title', 'authors']].iloc[start_idx:end_idx].astype(str).apply(lambda x: ' '.join(x), axis=1)
            batch_ids = df['bookID'].iloc[start_idx:end_idx].astype(str)
            batch_meta = df[['bookID', 'title', 'authors']].iloc[start_idx:end_idx].apply(lambda x: x.to_dict(), axis=1).tolist()

            # Embed batch texts
            batch_embeddings = embedding_model.embed_documents(batch_texts.tolist())
            
            for id, embedding, metadata in zip(batch_ids, batch_embeddings, batch_meta):
                vectors.append({
                    'id': str(id),
                    'values': embedding,
                    'metadata': metadata
                })

        if vectors:
            index.upsert(vectors=vectors)
            print(f"Data upserted to Pinecone. {len(vectors)} vectors processed.")
        else:
            print("No vectors to upsert.")
    
    except Exception as e:
        print(f"Error upserting data to Pinecone: {e}")
upsert_data()  # Ensure data is upserted before running the Streamlit app


def fetch_top_k_entries(query_text, k=3, threshold=0.5):
    try:
        # Embed the query text using Hugging Face model
        query_embedding = embedding_model.embed_documents([query_text])[0]

        # Query Pinecone index
        result = index.query(vector=query_embedding, top_k=k, include_metadata=True)

        if result and 'matches' in result and len(result['matches']) > 0:
            all_results = []
            for match in result['matches']:
                title = match['metadata'].get(text_key, 'No title found')
                authors = match['metadata'].get(authors_key, 'No authors found')

                # Check if the query_text matches title or authors
                if query_text.lower() in title.lower() or query_text.lower() in authors.lower():
                    all_results.append({
                        'title': title,
                        'authors': authors,
                        'metadata': match['metadata']
                    })
                else:
                    # Check for partial matching in authors
                    query_authors = query_text.lower().split()  # Split query authors by spaces
                    match_authors = authors.lower().split()  # Split matched authors by spaces
                    if all(query_author in match_authors for query_author in query_authors):
                        all_results.append({
                            'title': title,
                            'authors': authors,
                            'metadata': match['metadata']
                        })

            if all_results:
                return all_results
            else:
                return None  # Return None when no exact or partial matches found
        else:
            return None  # Return None when no results found

    except Exception as e:
        print(f"Error during fetching top-k entries: {e}")
        return None



def main():
    st.title("🦜🔗Book Recommendation Chatbot")

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    query_text = st.text_input("Enter your query:", key="query")
    k = 3
    if st.button("Get Recommendations"):
        results = fetch_top_k_entries(query_text, k)
        if results:
            st.session_state['history'].append({
                'query': query_text,
                'results': results
            })
        else:
            st.session_state['history'].append({
                'query': query_text,
                'results': "No data found about it."
            })

    if st.session_state['history']:
        for session in reversed(st.session_state['history']):
            if session['results'] == "No data found about it.":
                st.markdown(f"🤖 **Chatbot:** No data found about '{session['query']}'")
            else:
                for result in session['results']:
                    if 'metadata' in result and 'title' in result['metadata'] and 'authors' in result['metadata']:
                        title_and_authors = result['metadata']['title'] + " by " + result['metadata']['authors']
                        params = {
                            "max_length": 256,
                            "min_length": 8,
                            "no_repeat_ngram_size": 3,
                            "early_stopping": True,
                            "repetition_penalty": 3.5,
                            "length_penalty": 0.3,
                            "encoder_no_repeat_ngram_size": 3,
                            "num_beams": 4,
                        }
                        summarized_result = summarizer(title_and_authors, **params)
                        if summarized_result:
                            st.markdown(f"📝 **User:** {session['query']}")
                            st.markdown(f"🤖  **Title:** {result['title']}, **Authors:** {result['authors']}")
                           
                            st.markdown("---")

if __name__ == "__main__":
   main()
