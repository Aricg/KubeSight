#!/usr/bin/env python

import chromadb
import openai
import os
import requests


def filter_results_by_distance(results, threshold, max_results):
    """Filter and return results based on distance threshold and maximum number of results."""

    distances = results['distances'][0]

    # Debug prints for the lengths of lists
    print(f"Debug - Length of Distances: {len(distances)}")
    print(f"Debug - Length of Documents: {len(results['documents'][0])}")

    filtered_documents = []
    filtered_metadatas = []
    filtered_distances = []

    for idx, distance in enumerate(distances):
        # Check if idx is a valid index for documents and metadatas
        if idx >= len(results['documents'][0]):
            print(f"Warning: Skipping distance at index {idx} due to lack of corresponding document.")
            continue

        # If the distance is below the threshold, store the document, metadata, and distance
        if distance < threshold:
            filtered_documents.append(results['documents'][0][idx])
            filtered_metadatas.append(results['metadatas'][0][idx])
            filtered_distances.append(distance)

    # Limit by count:
    filtered_documents = filtered_documents[:max_results]
    filtered_metadatas = filtered_metadatas[:max_results]
    filtered_distances = filtered_distances[:max_results]

    return filtered_documents, filtered_metadatas, filtered_distances


def get_openai_embedding(text, model_id="text-embedding-ada-002", api_key=None):
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "input": text,
        "model": model_id
    }

#    print(f"Text sent for embedding: {text}")
    response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=data)
    return response.json()


def query_chromadb(query_text, n_results=10, metadata_filter=None, document_filter=None):
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="/tmp")

    # Get the collection
    collection = chroma_client.get_collection(name="kube_commands")

    # Get the embedding for the query text using the OpenAI model
    query_embedding_response = get_openai_embedding(query_text)

    # Debug: Print the embedding response from OpenAI.
    print("Embedding Response:")
    print(f"Vector Length: {len(query_embedding_response)}")

    # Extract the actual embedding vector from the response. 
    # Adjust the key based on the actual structure of the response.
    query_embedding_vector = query_embedding_response['data'][0]['embedding']

    # Debug: Print the extracted embedding vector.
    print(f"Extracted Query Embedding Vector:, {len(query_embedding_vector)}")

    # Perform the query using the embedded vector
    results = collection.query(
        query_embeddings=[query_embedding_vector],
        n_results=n_results,
        where=metadata_filter,
        where_document=document_filter,
        include=["metadatas", "documents", "distances"]
    )

    # Debug: Print the query results.
    print("Query Results:", results)

    return results


def display_chromadb_contents():
    """
    Display the contents of the ChromaDB database for debugging purposes.
    """

    # Initialize ChromaDB persistent client
    chroma_client = chromadb.PersistentClient(path="/tmp")

    collection = chroma_client.get_collection(name="kube_commands")

    # Use the peek() method to get the first few items from the collection
    items = collection.peek()

    # Iterate over the items and display the associated document and metadata
    for idx, doc_id in enumerate(items['ids']):
        print(f"ID: {doc_id}")
        print(f"Document: {items['documents'][idx]}")
        print(f"Metadata: {items['metadatas'][idx]}")
#        print(f"embeddings: {items['embeddings'][idx]}")
        print('-'*40)  # Separator for better readability


def display_results_for_index_0(results):
    # Check if we have any results
    if not results['ids']:
        print("No results found.")
        return

    # Extract details for index 0
    doc_id = results['ids'][0][0]  # Assuming ids are structured similarly to other fields
    document = results['documents'][0][0]
    metadata = results['metadatas'][0][0]
    distance = results['distances'][0][0]

    # Display the extracted details
    print(f"ID: {doc_id}")
    print(f"Document: {document}")
    print(f"Metadata: {metadata}")
    print(f"Distance: {distance}")
    print('-'*40)

# Display the contents of the database
display_chromadb_contents()

query="minecraft"
# Query the database
results = query_chromadb(query, n_results=100)

# Display the result at index 0
print("display_results_for_index_0")
display_results_for_index_0(results)

# Filter results by distance
filtered_documents, filtered_metadatas, filtered_distances = filter_results_by_distance(results, 0.7, 100)

print("#####################FILTERED SHOUD ONLY RETURN 1################################")
# Display the most relevant document (smallest distance)
if filtered_distances:
    min_distance_index = filtered_distances.index(min(filtered_distances))
    print(f"ID: Not available in current setup")  # You might need to adjust based on how you handle IDs
    print(f"Document: {filtered_documents[min_distance_index]}")
    print(f"Metadata: {filtered_metadatas[min_distance_index]}")
    print(f"Distance: {filtered_distances[min_distance_index]}")
    print('-'*40)
else:
    print("No documents found within the specified distance threshold.")



