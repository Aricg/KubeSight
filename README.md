#Kubesight
# Vector Embedding of YAML output

## Overview

This Python script processes yaml, optionally sends it to OpenAI for embedding, and stores the result in a ChromaDB database. It also provides an option to save the embedding to a file.

## Table of Contents

- [Getting Started](#getting-started)
  - [Dependencies](#dependencies)
  - [Usage](#usage)
- [Functions](#functions)
  - [command_already_embedded](#command_already_embedded)
  - [display_chromadb_contents](#display_chromadb_contents)
  - [compute_id_from_command](#compute_id_from_command)
  - [store_embedding_in_chromadb](#store_embedding_in_chromadb)
  - [save_embedding_to_disk](#save_embedding_to_disk)
  - [parse_helm_output](#parse_helm_output)
  - [get_openai_embedding](#get_openai_embedding)
  - [main](#main)

## Getting Started

### Dependencies

This script requires the following Python libraries: `sys`, `re`, `time`, `os`, `argparse`, `yaml`, `json`, `requests`, `subprocess`, `chromadb`, and `hashlib`.

### Usage

To run the script, use the following command:

```bash
python vector_embed_yaml.py [-p] [-e] [-s] [-f FILENAME] -c COMMAND
```

Where:

- `-p` or `--parse`: Only parse the Helm output.
- `-e` or `--embed`: Send the parsed Helm output to OpenAI for embedding.
- `-s` or `--save`: Save the embedding to a file.
- `-f` or `--filename`: Filename to save the embedding. A default is used if not provided.
- `-c` or `--command`: Command that outputs YAML to be processed.

At least one of the actions (`--parse` or `--embed`) must be chosen. The `--command` is required.

## Functions

### command_already_embedded

Checks if the given command is already in the metadata of any stored embeddings.

### display_chromadb_contents

Displays the contents of the ChromaDB database for debugging purposes.

### compute_id_from_command

Computes a unique hash-based ID from the given command.

### store_embedding_in_chromadb

Stores the embedding in ChromaDB with associated metadata.

### save_embedding_to_disk

Saves the embedding to a file.

### parse_helm_output

Parses the output from a Helm command.

### get_openai_embedding

Sends a text to OpenAI for embedding.

### main

The main idea of the script. Is to create a vector representation based on yaml configs/output for example all of the installed helm charts on a cluster. Then send it to OpenAI for embedding, and stores the result in a ChromaDB database. then we an build a knowledge base an debugging assisant.
