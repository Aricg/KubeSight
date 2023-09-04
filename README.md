# vector_embed_yaml.py Documentation

This script processes arbitrary yaml output (eg: helm or kubectl commands) sends it to OpenAI for text embedding and stores the embedding in a ChromaDB database.
Then we query the ChromaDB for similar documents based on the embedded vector which we can provide in the form of context when asking questions to OpenAI's GPT-4 model.

## Index
1. [Requirements](#requirements)
2. [Usage](#usage)
3. [Functions](#functions)
4. [Error Handling](#error-handling)

### Requirements
- Python 3.6 or above
- Libraries: sys, re, time, os, argparse, yaml, json, requests, subprocess, chromadb, hashlib, openai

### Usage
```
python vector_embed_yaml.py [-h] [-p] [-e] [-c COMMAND] [-q QUERY] [--n_results N_RESULTS]
```
- `-p` or `--parse`: Only parse the helm output.
- `-e` or `--embed`: Send the parsed helm output to OpenAI for embedding.
- `-c` or `--command`: Command that outputs YAML to be processed.
- `-q` or `--query`: Query the ChromaDB for similar documents.
- `--n_results`: Number of similar results to fetch. Default is 2.

### Functions
- `get_gpt4_response(embeddings, user_question, model="gpt-4")`: Fetches a response from the GPT-4 model based on the user's question and the document embeddings.
- `query_chromadb(query_text, n_results=10, metadata_filter=None, document_filter=None)`: Queries the ChromaDB for similar documents based on the embedded vector.
- `command_already_embedded(command, collection)`: Checks if the given command is already in the metadata of any stored embeddings.
- `display_chromadb_contents()`: Displays the contents of the ChromaDB database for debugging purposes.
- `compute_id_from_command(command)`: Computes a unique ID for each command using SHA-256 hashing.
- `store_embedding_in_chromadb(embedding_response, command, helm_output)`: Stores the embedding in ChromaDB with associated metadata.
- `parse_helm_output(output)`: Parses the helm output into separate documents.
- `get_openai_embedding(text, model_id="text-embedding-ada-002", api_key=None)`: Sends text to OpenAI for embedding.
- `main(args)`: Main function that processes command-line arguments and orchestrates the script's workflow.

### Error Handling
The script exits with an error message if:
- No action (parse or embed) is chosen.
- No helm command is provided.
