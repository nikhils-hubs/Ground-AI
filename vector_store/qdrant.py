import os 
import uuid
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (Distance, VectorParams, PointStruct, 
                                   Filter, FieldCondition, MatchValue) 
from langchain_core.documents import Document

load_dotenv()

class QdrantRepository:
    
    def __init__(self):
        self.client = AsyncQdrantClient(
            url = os.getenv("QDRANT_ENDPOINT"),
            api_key = os.getenv("QDRANT_API_KEY"),
        )
        self.collection_name = "ground-ai-vectors"
      
    async def create_collection(self):
        if await self.client.collection_exists(collection_name = self.collection_name):
            return 
        await self.client.create_collection(
            collection_name = self.collection_name,
            vectors_config = VectorParams(size = 1024, distance = Distance.COSINE)
        )
        
    async def insert(self, chunks: list[Document], embeddings):
        point = []
        for chunk, embedding in zip(chunks,embeddings):
            chunk_id = chunk.metadata.get("chunk_id") or str(uuid.uuid4())
            point.append(
                PointStruct(
                    id = chunk_id,
                    vector = embedding,
                    payload={
                        **chunk.metadata,
                        "chunk_id": chunk_id,
                        "text": chunk.page_content,
                    },
                )
            )
            await self.client.upsert(
                collection_name = self.collection_name,
                points = point
            )
                
    
    async def search(self, embedding, workplace_id, limit = 20):
        result = await self.client.query_points(
            collection_name = self.collection_name,
            query = embedding,
            query_filter = Filter(
                must = [FieldCondition(key = workplace_id, match = MatchValue(value = workplace_id))],
            ),
            limit = limit,
        )
        return result
        
    
    async def delete(self, document_id):
        await self.client.delete(
            collection_name = self.collection_name,
            points_selector = Filter(
                must = [FieldCondition(key = "document_id", match = MatchValue(value = document_id))]
            ),
        )
        return document_id
    
    async def get_document_chunks(self, document_id):
        result,_ = await self.client.scroll(
            collection_name = self.collection_name,
            scroll_filter = Filter(
                must = [FieldCondition(key = "document_id", match = MatchValue(value = document_id))]   
            ),
            limit = 100,
        )
        return result