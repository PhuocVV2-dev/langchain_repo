from typing import List, Optional
from langchain.embeddings import OpenAIEmbeddings


from dotenv import dotenv_values, find_dotenv

ENV = dotenv_values()

class OpenAIEmbeddingsCustom(OpenAIEmbeddings):
    
    openai_api_key = ENV["OPENAI_API_KEY"]
    chunk_size = 100
    
    # openai_api_base= ENV["OPENAI_API_BASE"]
    # openai_api_type= ENV["OPENAI_API_TYPE"]
    
    def embed_sentences(self, sentences: List[str]) -> List[List[float]]:
        embeddings = self.embed_documents(sentences)
        return embeddings


if __name__ == "__main__":
    # client = "https://api.openai.com/v1"
    # embeddings_model = OpenAIEmbeddingsCustom() 

    # sentences = [
    #     "What's your name?",
    #     "Somethings is wrong",
    # ]
    # embeddings = embeddings_model.embed_sentences(sentences)
    # #print(embeddings)
    # print(type(embeddings))
    # print(type(embeddings[0]))
    pass
