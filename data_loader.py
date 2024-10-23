# data_loader.py

import json
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def load_data(json_file):
    """
    Load JSON data from a file.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        list: List of dictionaries containing 'instruction' and 'response'.
    """
    print(f"Loading data from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records.")
    return data

def build_vector_store(data):
    """
    Build a FAISS vector store from the data.

    Args:
        data (list): List of dictionaries with 'instruction' and 'response'.

    Returns:
        FAISS: A FAISS vector store.
    """
    print("Building vector store...")
    # Extract instructions and responses
    instructions = [item['instruction'] for item in data]
    responses = [item['response'] for item in data]

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    # Build vector store
    vector_store = FAISS.from_texts(
        instructions,
        embeddings,
        metadatas=[{'response': resp} for resp in responses]
    )
    print("Vector store built successfully.")
    return vector_store

if __name__ == "__main__":
    # Load data
    data = load_data('/Users/rohitganti/Desktop/cudaq-assistant/aa1/data.json')

    # Build vector store
    vector_store = build_vector_store(data)

    # Save the vector store for later use
    print("Saving vector store to 'vector_store' directory...")
    vector_store.save_local('/Users/rohitganti/Desktop/cudaq-assistant/aa1/vector_store')
    print("Vector store saved successfully.")
