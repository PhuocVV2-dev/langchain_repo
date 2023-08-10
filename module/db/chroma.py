from dotenv import load_dotenv, find_dotenv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chromadb
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from open.embedding import OpenAIEmbeddingsCustom
load_dotenv(find_dotenv())


class ChromaPersistClient(Chroma):
    def __init__(self, persist_directory, collection_name: str, embedding_model: OpenAIEmbeddingsCustom):
        self.collection_name = collection_name
        self.embedding_function = embedding_model
        self.persist_directory = persist_directory
        self._client_settings = Settings(
            chroma_db_impl=os.environ.get('CHROMA_DB_IMPL'),
            persist_directory=self.persist_directory # Optional, defaults to .chromadb/ in the current directory
            )
        
    def get_collection_persist(self, path):
        return chromadb.PersistentClient(path=self.persist_directory).get_or_create_collection(self.collection_name)
    

class ChromaDB(Chroma):
    def __init__(self, client: ChromaPersistClient, collection_name: str,  embeding_model: OpenAIEmbeddingsCustom, persist_directory: str = 'db/db_v1'):
        self.client = client
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embeddings_model = embeding_model
        self.client_settings = Settings(
            chroma_db_impl=os.environ.get('CHROMA_DB_IMPL'),
            persist_directory=self.persist_directory # Optional, defaults to .chromadb/ in the current directory
            )
    def check_persist_dir_exist(self):
        if os.path.exists(self.persist_directory):
            print('Persist directory is not exist, Auto create')
            os.mkdir(self.persist_directory)

    def load_documents(self, doc_path: str = 'state_of_the_union.txt', chunk_size: int = 1000, chunk_overlap: int = 0):
        if not os.path.exists(doc_path):
            print("null")
            print(os.getcwd())
            pass
        loader = TextLoader(doc_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
        return texts

    def init_vectordb(self, doc_path:str):
        self.check_persist_dir_exist()
        texts = self.load_documents(doc_path, chunk_size=1000, chunk_overlap=0)
        vectordb = client.from_documents(
            documents=texts, embedding=self.embeddings_model, persist_directory=self.persist_directory)
        vectordb.persist()
        return vectordb

    def load_vectordb_from_persist(self):
        # Load data from persist dir
        
        vectordb = Chroma(embedding_function=self.embeddings_model,
                          persist_directory=self.persist_directory,
                          collection_name=self.collection_name,
                          client_settings=self.client_settings)
        return vectordb

    def search_query(self, query, k):
        vectordb = self.load_vectordb_from_persist()



        

if __name__ == "__main__":
    path_dir = 'db/db_v1'
    emd = OpenAIEmbeddingsCustom()
    client = ChromaPersistClient(persist_directory=path_dir, collection_name='restrict', embedding_model=emd)
    chromadb = ChromaDB(client=client, collection_name='restrict', embeding_model=emd, persist_directory=path_dir)
    vectordb = chromadb.init_vectordb(doc_path='state_of_the_union.txt')
    
    # TODO: OpenAIEmbedding -> ChromaPersistClient -> ChromaDB 