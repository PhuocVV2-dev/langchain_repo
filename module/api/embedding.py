from typing import List, Optional
from langchain.embeddings import OpenAIEmbeddings

""" 
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://<your-endpoint.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "your AzureOpenAI key"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_PROXY"] = "http://your-corporate-proxy:8080"
"""


class OpenAIEmbeddingsCustom(OpenAIEmbeddings):
    def __init__(self, client: str, openai_api_key: str, chunk_size=100):
        super().__init__(client=client, openai_api_key=openai_api_key, chunk_size=chunk_size)

    def embed_sentences(self, sentences: List[str]) -> List[List[float]]:
        embeddings = super().embed_documents(sentences)
        return embeddings


if __name__ == "__main__":
    client = "https://api.openai.com/v1"
    openai_api_key = "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"
    embeddings_model = OpenAIEmbeddingsCustom(
        client=client, openai_api_key=openai_api_key)

    sentences = [
        "What's your name?",
        "Somethings is wrong",
    ]
    embeddings = embeddings_model.embed_sentences(sentences)
    print(embeddings)
    print(type(embeddings))
    print(type(embeddings[0]))
