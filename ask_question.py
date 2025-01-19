import streamlit as st
from sentence_transformers import SentenceTransformer
import pinecone
import google.generativeai as genai
from pinecone import Pinecone

# Configure API keys and initialize clients
genai.configure(api_key="AIzaSyBVPq6QUM156sNEXpPDJaPycmUMdOHZfOo")
pc = Pinecone(
    api_key="pcsk_5Fk6cs_L8N9FbdByyJqFFHsr3J6K1RLBonzMav3jmzGY9TA18zave7MWkDDtnSWnp3Vj6r",
    environment="us-west1-gcp"
)

index_name = "hr-policy-index"
index = pc.Index(index_name)

# Initialize Sentence-Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate embeddings
def generate_embeddings(text):
    return model.encode(text, convert_to_tensor=False).tolist()

# Function to query Pinecone
def query_pinecone(query):
    query_embedding = generate_embeddings(query)
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    return results.get('matches', [])

# Function to generate content using the Gemini AI model
def generate_gemini_content(transcript_text, prompt):
    try:
        print(transcript_text)
        response = genai.GenerativeModel("gemini-pro").generate_content(prompt + transcript_text)
        return str(response._result.candidates[0].content.parts[0].text)
    except (KeyError, IndexError, AttributeError) as e:
        st.error(f"Error generating content: {e}")
        return None

# Function to handle query input
def handle_query():
    query = st.text_input("Enter your query about HR policies:")
    if query:
        results = query_pinecone(query)
        if results:
            st.write("Here is the most relevant HR policy content:")
            context = results[0]['metadata']['text']  # Get the relevant content
            answer = generate_gemini_content(context, query)
            if answer:
                st.write(f"Answer: {answer}")
            else:
                st.write("Failed to generate an answer.")
        else:
            st.write("No relevant information found.")

# Streamlit app UI
def app():
    st.title("HR Policy Query System")
    handle_query()

# Run the app
if __name__ == "__main__":
    app()