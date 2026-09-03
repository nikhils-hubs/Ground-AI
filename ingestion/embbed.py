from langchain_huggingface import HuggingFaceEmbeddings
from vector_store.qdrant import QdrantRepository

embedder = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-m3",
    encode_kwargs = {"normalize_embedding": True}
)

async def embed(chunks):
    embedding = embedder.embed_documents(
        [chunk.page_content for chunk in chunks]
    )
    repo = QdrantRepository()
    await repo.create_collection()
    await repo.insert(chunks, embedding)

