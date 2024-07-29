import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set your OPENAI_API_KEY in the environment variable directly or through your script initialization
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")

def get_source_data(file_path='RnJ_play.txt'):
    with open(file_path, 'r') as file:
        document_content = file.read()

    act_segments = document_content.split("ACT")
    act_segments = [segment.strip() for segment in act_segments if segment.strip()]
    return act_segments

def vectorize_texts(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

def query_vector(query, vectorizer):
    tfidf_query = vectorizer.transform([query])
    return tfidf_query

# Vectorize source data
source_texts = get_source_data()
vectorizer, tfidf_matrix_source = vectorize_texts(source_texts)

user_query = input("Learn more about Romeo and Juliet play: ")
tfidf_query = query_vector(user_query, vectorizer)

cosine_similarities = cosine_similarity(tfidf_query, tfidf_matrix_source).flatten()
most_similar_doc_indices = cosine_similarities.argsort()[::-1]

# Retrieve the most relevant document(s)
top_doc_indices = most_similar_doc_indices[:1]  # Adjust the number based on how many results you want
top_docs = [source_texts[i] for i in top_doc_indices]

print("Most relevant documents to the query:")
for doc in top_docs:
    print(doc)

# Note: The interaction with langchain_openai and handling SystemMessage, HumanMessage, AIMessage is omitted for clarity
