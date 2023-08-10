from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader

from dotenv import load_dotenv, find_dotenv
from module.open.embedding import OpenAIEmbeddingsCustom

import chromadb

ENV = load_dotenv(find_dotenv())


def extract_document():
    # Load and process the text
    loader = TextLoader('state_of_the_union.txt')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def initalizePersistedChromaDB(embeddings, texts, persist_directory):
    # Embed and store the texts
    # Supplying a persist_directory will store the embeddings on disk

    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb

# def load_vectordb(persist_client, persist_directory, embedding):
#     vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
#     return qa

if __name__ == '__main__':
    client = chromadb.PersistentClient(path="/db")
    embedding = OpenAIEmbeddingsCustom(client=client)
    texts = extract_document()
    vectordb = initalizePersistedChromaDB(embeddings=embedding, texts=texts, persist_directory='/db')
    #qa = load_vectordb(persist_client=client, persist_directory="/db", embedding=embedding)
    #query = "What did the president say about Ketanji Brown Jackson"
    #print(qa.run(query))
    