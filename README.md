🦜🔗

**Book Recommendation Chatbot**

This repository contains a Book Recommendation Chatbot built using Pinecone for vector indexing, LangChain for embedding and interaction management, Hugging Face models for NLP tasks, and Streamlit for the web interface. The chatbot provides personalized book suggestions based on user input queries.
---------------------------------------
***Features***
Personalized Book Recommendations: Suggest books based on user queries.
Vector Indexing: Efficient search and retrieval using Pinecone.
Advanced NLP Models: Utilize Hugging Face models for text embedding and generation.
Interactive Web Interface: User-friendly interface built with Streamlit.
Prerequisites
Before running the project, ensure you have the following installed:

Python 3.12.3
Pinecone
LangChain
Hugging Face Transformers
Streamlit
tqdm
torch
---------------------------------
***Installation***

1- Clone the repository:

```
git clone https://github.com/your-username/book-recommendation-chatbot.git
cd book-recommendation-chatbot
```
2-Create and activate a virtual environment:
```
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```
3- Install the required packages:
```
pip install -r requirements.txt
````
4-Set up Pinecone environment variables:
,,,
export PINECONE_API_KEY='your_pinecone_api_key'
export PINECONE_ENVIRONMENT='your_pinecone_environment'
,,,,

-------------------------------------------------
***Usage***

1- Ensure your dataset (books.csv) is in the project directory.

2- Run the Streamlit app:

```
streamlit run fine_name.py
```

3-Open your browser and navigate to http://localhost:8501 to interact with the chatbot.
-----------------------------------------------
***Code Explanation***

Pinecone Initialization:

Pinecone is initialized using an API key and environment settings. The code checks if the specified index exists and creates it if it doesn't. This index is configured to use cosine similarity for vector comparisons.

Data Preprocessing:

The dataset is loaded and cleaned by removing rows with null values in the 'title' or 'authors' columns. Text fields are converted to strings to ensure compatibility with the embedding model.

Embedding and Upserting Data:

A pre-trained Hugging Face model is used to embed book titles and authors into high-dimensional vectors. These embeddings are upserted into the Pinecone index in batches to handle large datasets efficiently.

Fetching and Displaying Recommendations:

User queries are embedded using the same Hugging Face model, and the top-k matches are retrieved from the Pinecone index. The results are filtered for exact and partial matches to ensure relevant recommendations.

Streamlit Interface:

Streamlit is used to create a web interface for the chatbot. Users can input queries and view recommendations. The interface maintains a history of user interactions, displaying previous queries and their corresponding results.
