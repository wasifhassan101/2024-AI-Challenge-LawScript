import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Add your OpenAI API key for authentication
OPENAI_API_KEY='your-api-key-here'

client = OpenAI(api_key=OPENAI_API_KEY)


# Function to load all CSV embeddings from a specified directory into memory
def load_embeddings(directory):
    embeddings = []
    sources = []
    contents = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, filename))
            if 'embedding' in df.columns:
                embeddings.extend(df['embedding'].apply(eval).tolist())
                sources.extend(df['source'].tolist())
                contents.extend(df['content'].tolist())
    return embeddings, sources, contents


# Function to find the top_k most similar items to the query embeddings
def find_most_similar(query_embeddings, embeddings, sources, contents, top_k=5):
    embeddings_array = np.array([np.array(emb) for emb in embeddings])
    similarities = cosine_similarity(query_embeddings, embeddings_array)
    best_indices = np.argsort(similarities[0])[::-1]
    return [(sources[idx], similarities[0][idx], contents[idx]) for idx in best_indices[:top_k]]


# Function to generate a response using OpenAI's API 
# based on provided content summary and user query
def generate_response(content_summary, query):
    prompt = f"Answer the question based on sections and caselaw:\n{content_summary}\n------\nQuestion: {query}"
    
    response = client.chat.completions.create(
        model= "gpt-3.5-turbo-0125",
        messages = [
        {
            "role": "system",
            "content": """You are a Pakistani AI Legal research assistant. 
                        You will take users query and answer on the basis of statute sections and caselaw provided.
                        You will limit your knowledge base to provided statute sections 
                        and caselaw only and in your answer 
                        you will cite the specific information by the section or article number 
                        or case citation or court citation.""",
        },
        {
            "role": "user",
            "content":prompt
        }
    ],
        stream = True
    )

    return response


# Function to get embeddings for a given query using OpenAI's API
def get_query_embeddings(query):
    embeddings_response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    
    embedding = embeddings_response.data[0].embedding
    # Reshape the embedding to make it 2D
    return np.array(embedding).reshape(1, -1)

def main():
    # Load embeddings once the application starts
    with st.spinner('Loading Data... Please wait.'):
        embeddings, sources, contents = load_embeddings('current_data')

    # Setting up the Streamlit interface
    st.title('LawScript AI')
    st.text("An AI Search Engine capable of answering legal queries.")
    st.text(" You can load more data into the tool from reserve_data to Embeddings folder.")
    query = st.text_input("Enter your legal query:", "")
    if st.button('Search'):
        if query:
            query_embeddings = get_query_embeddings(query)
            with st.spinner('Searching... Please wait.'):
                results = find_most_similar(query_embeddings, embeddings, sources, contents, 3)
            content_summary = ""
            for source, similarity, content in results:
                content_summary += f"{source}: {content}\n\n"

            # Generate response based on the search results
            if content_summary:
                with st.spinner('Generating Response... Please wait.'):
                    response = generate_response(content_summary, query)
                st.write("AI Generated Response:")
                st.write(response)
                st.markdown("**References:**")
                for source, similarity, content in results:
                    with st.expander(f"{source}"):
                        st.write(f"{content}")
        else:
            st.write("Please enter a query to search.")


if __name__ == "__main__":
    main()