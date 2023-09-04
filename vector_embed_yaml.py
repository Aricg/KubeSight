#!/usr/bin/env python

import sys
import os
import argparse
import yaml
import requests
import subprocess
import chromadb
import hashlib
import openai
import tiktoken


def filter_results_by_distance(results, threshold, max_results):
    """Filter and return results based on distance threshold and maximum number of results."""

    distances = results['distances'][0]

    # Add debug prints for the lengths of lists
    print(f"Debug - Length of Distances: {len(distances)}")
    print(f"Debug - Length of Documents: {len(results['documents'])}")

    filtered_documents = []
    filtered_metadatas = []
    filtered_distances = []

    for idx, distance in enumerate(distances):
        # Check if idx is a valid index for documents and metadatas
        if idx >= len(results['documents']):
            print(f"Warning: Skipping distance at index {idx} due to lack of corresponding document.")
            continue

        # If the distance is below the threshold, store the document, metadata, and distance
        if distance < threshold:
            filtered_documents.append(results['documents'][idx])
            filtered_metadatas.append(results['metadatas'][idx])
            filtered_distances.append(distance)

    # Limit by count:
    filtered_documents = filtered_documents[:max_results]
    filtered_metadatas = filtered_metadatas[:max_results]
    filtered_distances = filtered_distances[:max_results]

    return filtered_documents, filtered_metadatas, filtered_distances


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_gpt4_response(embeddings, user_question, model="gpt-4"):
    # Convert the embeddings to a string format that provides meaningful context.
    context = f"Based on the document '{embeddings['documents'][0]}' with distance {embeddings['distances'][0]}, "
    message_content = context + user_question
    
    print("Sending the following content to OpenAI: ", message_content)

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": message_content}]
    )
    
    return completion.choices[0].message['content']




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



def command_already_embedded(command, collection):
    """
    Check if the given command is already in the metadata of any stored embeddings.

    Args:
    - command (str): The command string.
    - collection: The ChromaDB collection.

    Returns:
    - bool: True if the command is already embedded, False otherwise.
    """
    items = collection.peek()
    for metadata in items['metadatas']:
        if metadata['command'] == command:
            return True
    return False



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

def compute_id_from_command(command):
    return hashlib.sha256(command.encode()).hexdigest()


def store_embedding_in_chromadb(embedding_response, command, helm_output):
    """
    Store the embedding in ChromaDB with associated metadata.

    Args:
    - embedding (dict): The embedding to store.
    - command (str): The command used to get the helm output.

    Returns:
    - None
    """
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="/tmp")

    # Get or create the collection
    collection = chroma_client.get_or_create_collection(name="kube_commands")

    # Compute unique ID for this command
    unique_id = compute_id_from_command(command)
    #print(unique_id)

    # Check if the embedding with this ID already exists
    existing_items = collection.peek()
    #print(existing_items['ids'])

    if unique_id in existing_items['ids']:
        print(f"Already embeded ID: {unique_id}")
        return  # Exit the function, skip adding the embedding

    # Get embedding vector from the response
    embedding_vector = embedding_response['data'][0]['embedding']
    print(f"Vector Length: {len(embedding_vector)}")
    print(f"First 10 values: {embedding_vector[:10]}")

    # Create metadata for the embedding
    metadata = {"command": command}
    print(f"Storing metadata: {metadata}")

    # Insert the embedding and metadata into the collection
    collection.add(
        embeddings=[embedding_vector],
        documents=[helm_output],  
        metadatas=[metadata],
        ids=[unique_id]
    )
    print("New embedding added")


def parse_yaml_output(output):
    # Splitting the output by document start '---'
    documents = output.split('---')
    # Removing any empty strings in the list
    documents = [doc.strip() for doc in documents if doc.strip()]
    # Parsing each document using PyYAML
    parsed_data = [yaml.safe_load(doc) for doc in documents]
    return parsed_data


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

    # Calculate number of tokens
    num_tokens = num_tokens_from_string(text)
    print(f"Number of tokens in the input string: {num_tokens}")

    # Optionally, you can decide to refuse strings with too many tokens
    #if num_tokens > SOME_LIMIT:
    #    raise ValueError(f"Input string has too many tokens ({num_tokens}), refusing to process.")

#    print(f"Text sent for embedding: {text}")
    response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=data)
    return response.json()


def main(args):
    process = subprocess.Popen(args.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    helm_output, error = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {error.decode()}")
        sys.exit(1)
    helm_output = helm_output.decode()

    parsed_helm_documents = parse_yaml_output(helm_output)

    # If only parse mode, print the parsed documents
    if args.parse:
        for doc in parsed_helm_documents:
            print(doc)
            print('-'*40)  # Separator for better readability
        return

    # Initialize ChromaDB persistent client
    chroma_client = chromadb.PersistentClient(path="/tmp")
#    collection = chroma_client.get_or_create_collection(name="kube_commands")


    # If the command is already embedded, skip fetching the embedding
#    if command_already_embedded(args.command, collection):
#        print(f"Embedding for command '{args.command}' already exists.")
#        display_chromadb_contents()
#        return

    # If embed mode, send the parsed output to OpenAI and print the response
    if args.embed:
        parsed_output = "\n---\n".join([yaml.dump(doc) for doc in parsed_helm_documents])
        embedding_response = get_openai_embedding(parsed_output)
        #print(embedding_response)
        store_embedding_in_chromadb(embedding_response, args.command, helm_output)

    display_chromadb_contents()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process helm output and optionally send to OpenAI for embedding.')
    parser.add_argument('-p', '--parse', action='store_true', help='Only parse the helm output.')
    parser.add_argument('-e', '--embed', action='store_true', help='Send the parsed helm output to OpenAI for embedding.')
    parser.add_argument('-c', '--command', type=str, help='Command that outputs YAML to be processed.')
    parser.add_argument('-q', '--query', type=str, help='Query the ChromaDB for similar documents.')
    parser.add_argument('--n_results', type=int, default=2, help='Number of similar results to fetch.')

    args = parser.parse_args()
    # Check if at least one of the actions (parse or embed) is chosen



    if args.query:

        display_chromadb_contents()
        results = query_chromadb(args.query, n_results=100)  # Query more than necessary, then filter

        # Filter the results
        DISTANCE_THRESHOLD = 0.7  # Adjust this value as needed
        docs, metas, dists = filter_results_by_distance(results, DISTANCE_THRESHOLD, args.n_results)

        combined_text = ""

        for idx, doc_id in enumerate(docs):
            # Ensure you're accessing individual documents and not the entire list
            document = docs[idx][0] if isinstance(docs[idx], list) else docs[idx]
            
            print(f"ID: {doc_id}")
            print(f"Metadata: {metas[idx]}")
            print(f"Distance: {dists[idx]}")
            print('-'*40)
        
            combined_text += f"{document} {metas[idx]} {dists[idx]} "


        num_tokens = num_tokens_from_string(combined_text)
        print(f"Total tokens in the combined text: {num_tokens}")

        # Ask user for a question
        user_question = input("Please enter your question for the expert system: ")
        combined_text += user_question

        # Count the tokens of the combined text
        num_tokens = num_tokens_from_string(combined_text)
        print(f"Total tokens in the combined text: {num_tokens}")

        # Check token limit
        if num_tokens > 2048:
            print("Error: Combined text exceeds token limit for the model.")
            sys.exit(1)

        # Fetch response from GPT-4
        gpt4_response = get_gpt4_response(docs, user_question)  # Note: You might need to modify the get_gpt4_response function to accept the filtered docs
        print(f"Expert System Response: {gpt4_response}")

    if not (args.parse or args.embed):
        print("Error: Please choose at least one action (parse or embed).")
        sys.exit(1)

    if not args.command:
        print("Error: Please provide a helm command that outputs yaml")
        sys.exit(1)

    main(args)
