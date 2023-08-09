# Automate ingres data from pdf, csv to some database (chromaDB)

## Step 1: Prepare storing pdf, csv file from data sources

- Data sources: S3, Blob storage, Google storage

## Step 2: Text embedding

Embeddings used to represent data to vector. Basic idea how data structures manage to represent words and longer text in a way that conveys their meaning.

Text embeddings: vector is an ordered sequence of numbers then can be apply compute between some pieces to each orther.

The number of values in a text embedding - dimension (depends on the embedding technique). Two types of embedding is sparse and dense.

Sparse vs dense

- Sparse: depends on the size of the vocabulary, compute quickly, context-free, no notion of semantics
- Dense: fixed size, longer computation, encode information, encode semantics. The dimensions of vectors represent different properties - about semantics and syntactic charactistics of words

Measuring the results

- Compare two vectors: apply general metrics for calculating the distance between them
- Distance: cosine similarity or dot product

## Step 1: Ingesting the Data into Vector Store (ChromaDB)

```py
# Load the pdf
pdf_path = "wiki_data_short.pdf"
loader = PDFPlumberLoader(pdf_path)
documents = loader.load()

# Split documents and create text snippets
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10, encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
texts = text_splitter.split_documents(texts)

persist_directory = config["persist_directory"]
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory
```

## References

### ChromaDB

**Chroma is the open-source embedding database**. Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs.
Chroma gives you the tools to:

- store embeddings and their metadata
- embed documents and queries
- search embeddings

Chroma prioritizes:

- simplicity and developer productivity
- analysis on top of search
- it also happens to be very quick

Create Chroma Client
Creae Chroma Collection

Changing the distance function:

Set chunk size to 1000 (the size refers to number of tokens, not characters, so this is roughly 4KB of text. The best chunking is dependent on the data you are dealing with) 

```py
collection = client.create_collection(
      name="collection_name",
      metadata={"hnsw:space": "cosine"} # l2 is the default,  "l2", "ip, "or "cosine".
  )
```
