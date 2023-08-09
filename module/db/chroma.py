from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from api.embedding import OpenAIEmbeddingsCustom

class ChromaDB(Chroma):
    def __init__(self, client, collection_name, openai_api_key):
        self.client = client
        self.openai_api_key = openai_api_key
        self.collection_name = collection_name
        self.embeddings_model = OpenAIEmbeddingsCustom(
            client=self.client, openai_api_key=self.openai_api_key)

    def embed_documents(self, documents):
        return self.embeddings_model.embed_documents(documents)
