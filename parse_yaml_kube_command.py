#!/usr/bin/env python


import sys
import re
import time
import os
import argparse
import yaml
import json
import requests
import subprocess
import chromadb
import hashlib


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
        print('-'*40)  # Separator for better readability



def compute_id_from_command(command):
    """
    Compute a unique hash-based ID from the given command.

    Args:
    - command (str): The command string.

    Returns:
    - str: A hash-based unique ID.
    """
    return hashlib.sha256(command.encode()).hexdigest()


def store_embedding_in_chromadb(embedding_response, command):
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

    # Check if the embedding with this ID already exists
    existing_items = collection.peek()
    if unique_id in existing_items['ids']:
        print(f"Add of existing embedding ID: {unique_id}")
        return  # Exit the function, skip adding the embedding

    # Get embedding vector from the response
    embedding_vector = embedding_response['data'][0]['embedding']

    # Create metadata for the embedding
    metadata = {"command": command}

    # Insert the embedding and metadata into the collection
    collection.add(
        embeddings=[embedding_vector],
        documents=[command],  
        metadatas=[metadata],
        ids=[unique_id]
    )


def save_embedding_to_disk(embedding, filename=None):
    with open(filename, 'w') as f:
        json.dump(embedding, f)

    return filename

def parse_helm_output(output):
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

    response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=data)
    return response.json()


def main(args):
    process = subprocess.Popen(args.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    helm_output, error = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {error.decode()}")
        sys.exit(1)
    helm_output = helm_output.decode()

    parsed_helm_documents = parse_helm_output(helm_output)

    # If only parse mode, print the parsed documents
    if args.parse:
        for doc in parsed_helm_documents:
            print(doc)
            print('-'*40)  # Separator for better readability

    # Initialize ChromaDB persistent client
    chroma_client = chromadb.PersistentClient(path="/tmp")
    collection = chroma_client.get_or_create_collection(name="kube_commands")

    # If the command is already embedded, skip fetching the embedding
    if command_already_embedded(args.command, collection):
        print(f"Embedding for command '{args.command}' already exists.")
        display_chromadb_contents()
        return

    # If embed mode, send the parsed output to OpenAI and print the response
    if args.embed:
        parsed_output = "\n---\n".join([yaml.dump(doc) for doc in parsed_helm_documents])
        embedding_response = get_openai_embedding(parsed_output)
        print(embedding_response)

        if args.save:
            saved_filename = save_embedding_to_disk(embedding_response, args.filename or "foo")
            print(f"Embedding saved to {saved_filename}")

        # Store the embedding in ChromaDB
        store_embedding_in_chromadb(embedding_response, args.command)
        display_chromadb_contents()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process helm output and optionally send to OpenAI for embedding.')
    parser.add_argument('-p', '--parse', action='store_true', help='Only parse the helm output.')
    parser.add_argument('-e', '--embed', action='store_true', help='Send the parsed helm output to OpenAI for embedding.')
    parser.add_argument('-s', '--save', action='store_true', help='Save the embedding to a file.')
    parser.add_argument('-f', '--filename', type=str, help='Filename to save the embedding. A default is used if not provided.')
    parser.add_argument('-c', '--command', type=str, help='Command that outputs YAML to be processed.')

    args = parser.parse_args()
    # Check if at least one of the actions (parse or embed) is chosen
    if not (args.parse or args.embed):
        print("Error: Please choose at least one action (parse or embed).")
        sys.exit(1)

    if not args.command:
        print("Error: Please provide a helm command that outputs yaml")
        sys.exit(1)

    main(args)
