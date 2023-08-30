#!/usr/bin/env python

import os
import argparse
import subprocess
import chromadb
from collections import namedtuple
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Define the named tuple for wrapping the manifests
Document = namedtuple('Document', ['page_content', 'metadata'])

# Ensure the OpenAI API key is available as an environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

def run_command(command):
    """Execute a shell command and return its output."""
    result = subprocess.run(command, stdout=subprocess.PIPE, shell=True, check=True)
    return result.stdout.decode()


class HelmChartQuery:

    COLLECTION_NAME = "helm_chart_embeddings"

    def __init__(self):
        self.helm_releases = self.get_helm_releases()
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.retriever = self.store_in_chroma()
        self.genie = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=self.retriever)
        self.chroma_client = chromadb.PersistentClient(path="/tmp/chroma.sqlite3")  # Using /tmp for simplicity
        self.ensure_collection_exists()


    def ensure_collection_exists(self):
        """
        Ensures the collection exists. If not, it creates one.
        """
        try:
            self.chroma_client.get_collection(name=self.COLLECTION_NAME)
        except ValueError:
            self.chroma_client.create_collection(name=self.COLLECTION_NAME)

    def get_helm_releases(self):
        # Fetch only the first helm chart
        return [run_command("helm list -q").splitlines()[0]]

    def store_in_chroma(self):
        documents = [Document(page_content=run_command(f"helm get manifest {release}"), metadata={"release": release}) for release in self.helm_releases]
        vectordb = Chroma.from_documents(documents, self.embeddings)
        return vectordb.as_retriever()

    def query(self, query_text):
        return self.genie.run(query_text)


#    def list_db(self):
#    # List the contents of the database
#        try:
#            collection = self.chroma_client.get_collection(name=self.COLLECTION_NAME)
#            items = collection.peek()
#            for item in items:
#                print(item)
#        except ValueError as e:
#            print(f"Error: {e}")
#

    def list_db(self):
        # List the contents of the database
        collection = self.chroma_client.get_collection(name=self.COLLECTION_NAME)
        items = collection.peek()  # This fetches the first 10 items in the collection
        print("First Item:", items)  # Debug print

        
       # for item in items:
       #     print("ID:", item['id'])
        #    print("Embedding:", item['embedding'])
         #   print("Metadata:", item['metadata'])
          #  print("Document:", item['document'])
           # print('-' * 50)  # Separator for readability
    
    def delete_db(self):
        # Delete or clear the database
        try:
            self.chroma_client.delete_collection(name=self.COLLECTION_NAME)
            print(f"Deleted collection '{self.COLLECTION_NAME}'")
        except ValueError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Helm Chart Query Tool')
    parser.add_argument('-l', '--list', action='store_true', help='List the database')
    parser.add_argument('-d', '--delete', action='store_true', help='Delete the database')
    parser.add_argument('-q', '--query', type=str, help='Query the database')

    args = parser.parse_args()

    helm_query = HelmChartQuery()

    if args.list:
        helm_query.list_db()
    elif args.delete:
        helm_query.delete_db()
    elif args.query:
        results = helm_query.query(args.query)
        print(results)
    else:
        print("No action specified. Use -h for help.")

