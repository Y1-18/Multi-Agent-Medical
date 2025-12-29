import os
import re
import logging
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.documents import Document
from langchain.storage import InMemoryStore, LocalFileStore
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams, OptimizersConfigDiff

class VectorStore:
    """
    Create vector store, ingest documents, retrieve relevant documents
    """
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.collection_name = config.rag.collection_name
        self.embedding_dim = config.rag.embedding_dim
        self.distance_metric = config.rag.distance_metric
        self.embedding_model = config.rag.embedding_model
        self.retrieval_top_k = config.rag.top_k
        self.vector_search_type = config.rag.vector_search_type
        self.vectorstore_local_path = config.rag.vector_local_path
        self.docstore_local_path = config.rag.doc_local_path

        self.client = QdrantClient(path=self.vectorstore_local_path)

    def _does_collection_exist(self) -> bool:
        """Check if the collection already exists in Qdrant."""
        try:
            collection_info = self.client.get_collections()
            collection_names = [collection.name for collection in collection_info.collections]
            return self.collection_name in collection_names
        except Exception as e:
            self.logger.error(f"Error checking for collection existence: {e}")
            return False

    def _create_collection(self):
        """Create a new collection with dense and sparse vectors."""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                },
            )
            self.logger.info(f"Created new collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            raise e
            
    def load_vectorstore(self) -> Tuple[QdrantVectorStore, LocalFileStore]:
        """Load existing vectorstore and docstore."""
        if not self._does_collection_exist():
            self.logger.error(f"Collection {self.collection_name} does not exist.")
            raise ValueError(f"Collection {self.collection_name} does not exist")
            
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        qdrant_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        
        docstore = LocalFileStore(self.docstore_local_path)
        return qdrant_vectorstore, docstore

    def create_vectorstore(
            self,
            document_chunks: List[str],
            document_path: str,
        ) -> Tuple[QdrantVectorStore, LocalFileStore, List[str]]:
        """
        Create a vector store from document chunks or upsert documents to existing store.
        """
        doc_ids = [str(uuid4()) for _ in range(len(document_chunks))]
        
        langchain_documents = []
        for id_idx, chunk in enumerate(document_chunks):
            langchain_documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": os.path.basename(document_path),
                        "doc_id": doc_ids[id_idx],
                        "source_path": os.path.join("http://localhost:8000/", document_path)
                    }
                )
            )
        
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        if not self._does_collection_exist():
            self._create_collection()
        
        qdrant_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        
        docstore = LocalFileStore(self.docstore_local_path)
        
        # Ingest documents
        qdrant_vectorstore.add_documents(documents=langchain_documents, ids=doc_ids)
        encoded_chunks = [chunk.encode('utf-8') for chunk in document_chunks]
        docstore.mset(list(zip(doc_ids, encoded_chunks)))

        # FIX: Added missing return statement
        return qdrant_vectorstore, docstore, doc_ids

    def retrieve_relevant_chunks(
            self,
            query: str,
            vectorstore: QdrantVectorStore,
            docstore: LocalFileStore,
        ) -> List[Dict[str, Any]]: # Note: Updated return type to match your logic
        """
        Retrieve relevant chunks based on a query.
        """
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=self.retrieval_top_k
        )
        
        retrieved_docs = []
        for chunk, score in results:
            doc_content_bytes = docstore.mget([chunk.metadata['doc_id']])[0]
            if doc_content_bytes:
                doc_content = doc_content_bytes.decode('utf-8')
                
                doc_dict = {
                    "id": chunk.metadata['doc_id'],
                    "content": doc_content,
                    "score": score,
                    "source": chunk.metadata['source'],
                    "source_path": chunk.metadata['source_path'],
                }
                retrieved_docs.append(doc_dict)
        
        return retrieved_docs