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


    # If embed mode, send the parsed output to OpenAI and print the response
    if args.embed:
        parsed_output = "\n---\n".join([yaml.dump(doc) for doc in parsed_helm_documents])
        embedding_response = get_openai_embedding(parsed_output)
        print(embedding_response)

        # Determine the filename
        if not args.filename:
            filename = "foo"
        else:
            filename = args.filename


        if args.save:
            saved_filename = save_embedding_to_disk(embedding_response, filename)
            print(f"Embedding saved to {saved_filename}")



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
