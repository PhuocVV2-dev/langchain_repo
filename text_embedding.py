
from langchain.embeddings import OpenAIEmbeddings
from numpy import dot
from numpy.linalg import norm

OPENAI_API_KEY = "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"

embeddings_model = OpenAIEmbeddings(client=any, openai_api_key=OPENAI_API_KEY)

embeddings = embeddings_model.embed_documents(
    [
        "What's your name?",
        "Somethings is wrong",
    ]
)

a = embeddings[0]
b = embeddings[1]

cos_sim = dot(a, b)/(norm(a)*norm(b))
print(cos_sim)
