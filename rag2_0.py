#helo 

import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import (SystemMessage,HumanMessage,AIMessage)
from sklearn.feature_extraction.text import TfidfVectorizer as vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# run> export OPENAI_API_KEY = ---     in cmd
# $env:OPENAI_API_KEY='your_actual_openai_api_key_here'    in powershell


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY"
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo'
)
def extract_embeddings(response):
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)

file_path = 'RnJ_play.txt'

with open(file_path, 'r') as file:
    document_conent = file.read()
    #documents = [line.strip() for line in file.readlines()]rag2_0.py

act_segments = document_conent.split("Scene")
act_segments = [segment.strip() for segment in act_segments if segment.strip()]

#print(act_segments)

client = OpenAI()

source_embeddings = client.embeddings.create(
    input=act_segments,
    model="text-embedding-ada-002"
)
print(len(source_embeddings.data[0].embedding))

#print(type(source_vectors.data[0].embedding))

messages = [
    SystemMessage(content="Hello world"),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?")
]


#user_query = [input("Learn more about Romeo and Juliet play: -------------->>>>>>")]
user_query = "When did Romeo and Juliet meet"

query_embedding = client.embeddings.create(
    input=user_query,
    model="text-embedding-ada-002"
)
print("Printing input prompt data vectors  :---------------------------------------------------")

print(len(query_embedding.data[0].embedding))
source_embeddings = extract_embeddings(source_embeddings)
query_embedding = extract_embeddings(query_embedding).flatten()

source_embeddings_norm = source_embeddings / np.linalg.norm(source_embeddings, axis=1)[:, np.newaxis]
query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)

# Calculate cosine similarities
cosine_similarities = np.dot(source_embeddings_norm, query_embedding_norm)

most_similar_index = np.argmax(cosine_similarities)

# Retrieve the most relevant segment
most_relevant_segment = act_segments[most_similar_index]

print("Most relevant segment to the query:")
print(most_relevant_segment)

new_source_knowledge = "\n".join(most_relevant_segment)
#print(new_source_knowledge)
augmented_prompt = f"""Using the contexts below, answer the query.

Contexts:
{new_source_knowledge}

Query: {user_query}"""

print("Answering based on input messages :--------------------------------------------------")

prompt = HumanMessage(content= augmented_prompt)
#prompt_vector  = query_vector(user_query)
messages.append(prompt)

res = chat(messages)
print(res.content)

messages.append(res)