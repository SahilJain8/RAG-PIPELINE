# Retrieval-Augmented Generation (RAG) Model for Question Answering

This repository contains code for implementing a Retrieval-Augmented Generation (RAG) model to answer questions based on contextual information from Stephen Hawking's "A Brief History of Time" and "The Universe in a Nutshell".

## Overview

The project utilizes the following components:

- **Dataset Preparation**: Text excerpts from the aforementioned books are used to create a dataset containing text chunks along with their embeddings. FAISS is integrated for efficient vector search within the dataset.

- **Model Setup**: Two powerful models are utilized:
  - **Embedding Model**: `all-mpnet-base-v2` is used for encoding text into embeddings.
  - **Generative Model**: The `lama` model is employed for language modeling and response generation within the RAG framework.

- **Query and Retrieval**: Users can input queries related to the content of the books. FAISS facilitates fast and efficient vector search to retrieve relevant text chunks from the dataset.

- **Prompt Creation**: Prompts for the RAG model are created by combining user queries with retrieved text chunks as context. This prompts the model to generate accurate and informative responses.

## Usage

1. **Clone the Repository**:

```bash
git clone https://github.com/SahilJain8/RAG-PIPELINE
cd RAG-PIPELINE
```

2. **Install requirements**
```Bash
pip install -r requirements.txt
```


2. **Run the project**
```Bash
python main.py
```

# Credits
This project is inspired by the works of Stephen Hawking and builds upon research in natural language processing. The RAG model implementation leverages open-source libraries and frameworks.