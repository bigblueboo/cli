#!/usr/bin/env python3
"""
vector_search - A self-documenting CLI tool for semantic document search.

This tool provides semantic search capabilities using various vector database
backends (ChromaDB, Pinecone, Weaviate, Qdrant) with automatic embedding
generation using sentence-transformers.

Exit Codes:
    0 - Success
    1 - Collection not found
    2 - Backend error (connection, API, etc.)
    3 - Invalid arguments

Environment Variables:
    VECTOR_SEARCH_DIR     - Storage directory (default: ~/.vector_search/)
    PINECONE_API_KEY      - Pinecone API key (required for Pinecone backend)
    WEAVIATE_URL          - Weaviate server URL (default: http://localhost:8080)
    QDRANT_URL            - Qdrant server URL (default: http://localhost:6333)

Examples:
    # Index documents
    vector_search index "docs/*.md" --collection knowledge-base
    vector_search index document.txt --collection my-docs
    vector_search index README.md --collection docs --backend chromadb

    # Query for similar documents
    vector_search query "How do I configure authentication?" --collection knowledge-base
    vector_search query "deployment steps" --collection my-docs --top-k 10

    # Collection management
    vector_search list-collections
    vector_search stats --collection knowledge-base
    vector_search delete --collection old-docs
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Exit codes
EXIT_SUCCESS = 0
EXIT_COLLECTION_NOT_FOUND = 1
EXIT_BACKEND_ERROR = 2
EXIT_INVALID_ARGS = 3


def get_storage_dir() -> Path:
    """Get the storage directory for vector databases."""
    storage_dir = os.environ.get("VECTOR_SEARCH_DIR", os.path.expanduser("~/.vector_search"))
    path = Path(storage_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_doc_id(content: str, source: str) -> str:
    """Generate a unique document ID from content and source."""
    combined = f"{source}:{content[:1000]}"
    return hashlib.md5(combined.encode()).hexdigest()


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model."""

    _instance = None
    _model = None

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
                self._model_name = model_name
            except ImportError:
                print("Error: sentence-transformers not installed. Run: pip install sentence-transformers", file=sys.stderr)
                sys.exit(EXIT_BACKEND_ERROR)
            except Exception as e:
                print(f"Error loading embedding model: {e}", file=sys.stderr)
                sys.exit(EXIT_BACKEND_ERROR)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings."""
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._model.get_sentence_embedding_dimension()


class VectorBackend(ABC):
    """Abstract base class for vector database backends."""

    @abstractmethod
    def create_collection(self, name: str, dimension: int) -> None:
        """Create a new collection."""
        pass

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        pass

    @abstractmethod
    def list_collections(self) -> List[str]:
        """List all collections."""
        pass

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        pass

    @abstractmethod
    def add_documents(self, collection: str, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """Add documents to a collection."""
        pass

    @abstractmethod
    def query(self, collection: str, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Query similar documents."""
        pass

    @abstractmethod
    def get_stats(self, collection: str) -> Dict[str, Any]:
        """Get collection statistics."""
        pass


class ChromaDBBackend(VectorBackend):
    """ChromaDB vector database backend."""

    def __init__(self):
        try:
            import chromadb
            storage_path = get_storage_dir() / "chromadb"
            storage_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(storage_path))
        except ImportError:
            print("Error: chromadb not installed. Run: pip install chromadb", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def create_collection(self, name: str, dimension: int) -> None:
        try:
            self.client.get_or_create_collection(name=name, metadata={"dimension": dimension})
        except Exception as e:
            print(f"Error creating collection: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def delete_collection(self, name: str) -> None:
        try:
            self.client.delete_collection(name=name)
        except Exception as e:
            if "does not exist" in str(e).lower():
                print(f"Collection '{name}' not found", file=sys.stderr)
                sys.exit(EXIT_COLLECTION_NOT_FOUND)
            print(f"Error deleting collection: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def list_collections(self) -> List[str]:
        try:
            collections = self.client.list_collections()
            return [c.name for c in collections]
        except Exception as e:
            print(f"Error listing collections: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def collection_exists(self, name: str) -> bool:
        return name in self.list_collections()

    def add_documents(self, collection: str, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        try:
            coll = self.client.get_or_create_collection(name=collection)
            ids = [doc["id"] for doc in documents]
            texts = [doc["content"] for doc in documents]
            metadatas = [{"source": doc["source"], "chunk_index": doc.get("chunk_index", 0)} for doc in documents]
            coll.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        except Exception as e:
            print(f"Error adding documents: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def query(self, collection: str, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        try:
            coll = self.client.get_collection(name=collection)
            results = coll.query(query_embeddings=[embedding], n_results=top_k, include=["documents", "metadatas", "distances"])

            output = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    output.append({
                        "content": doc,
                        "source": results["metadatas"][0][i].get("source", "unknown") if results["metadatas"] else "unknown",
                        "score": 1 - results["distances"][0][i] if results["distances"] else 0,  # Convert distance to similarity
                    })
            return output
        except Exception as e:
            if "does not exist" in str(e).lower():
                print(f"Collection '{collection}' not found", file=sys.stderr)
                sys.exit(EXIT_COLLECTION_NOT_FOUND)
            print(f"Error querying collection: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def get_stats(self, collection: str) -> Dict[str, Any]:
        try:
            coll = self.client.get_collection(name=collection)
            count = coll.count()
            metadata = coll.metadata or {}
            return {
                "name": collection,
                "document_count": count,
                "dimension": metadata.get("dimension", "unknown"),
                "backend": "chromadb",
            }
        except Exception as e:
            if "does not exist" in str(e).lower():
                print(f"Collection '{collection}' not found", file=sys.stderr)
                sys.exit(EXIT_COLLECTION_NOT_FOUND)
            print(f"Error getting stats: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)


class PineconeBackend(VectorBackend):
    """Pinecone vector database backend."""

    def __init__(self):
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            print("Error: PINECONE_API_KEY environment variable not set", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

        try:
            from pinecone import Pinecone, ServerlessSpec
            self.pc = Pinecone(api_key=api_key)
            self.ServerlessSpec = ServerlessSpec
        except ImportError:
            print("Error: pinecone not installed. Run: pip install pinecone", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)
        except Exception as e:
            print(f"Error connecting to Pinecone: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def _sanitize_name(self, name: str) -> str:
        """Pinecone index names must be lowercase alphanumeric with hyphens."""
        return name.lower().replace("_", "-")

    def create_collection(self, name: str, dimension: int) -> None:
        name = self._sanitize_name(name)
        try:
            if name not in [idx.name for idx in self.pc.list_indexes()]:
                self.pc.create_index(
                    name=name,
                    dimension=dimension,
                    metric="cosine",
                    spec=self.ServerlessSpec(cloud="aws", region="us-east-1")
                )
        except Exception as e:
            print(f"Error creating index: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def delete_collection(self, name: str) -> None:
        name = self._sanitize_name(name)
        try:
            self.pc.delete_index(name)
        except Exception as e:
            if "not found" in str(e).lower():
                print(f"Collection '{name}' not found", file=sys.stderr)
                sys.exit(EXIT_COLLECTION_NOT_FOUND)
            print(f"Error deleting index: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def list_collections(self) -> List[str]:
        try:
            return [idx.name for idx in self.pc.list_indexes()]
        except Exception as e:
            print(f"Error listing indexes: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def collection_exists(self, name: str) -> bool:
        name = self._sanitize_name(name)
        return name in self.list_collections()

    def add_documents(self, collection: str, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        collection = self._sanitize_name(collection)
        try:
            index = self.pc.Index(collection)
            vectors = []
            for doc, emb in zip(documents, embeddings):
                vectors.append({
                    "id": doc["id"],
                    "values": emb,
                    "metadata": {
                        "content": doc["content"][:1000],  # Pinecone metadata size limit
                        "source": doc["source"],
                        "chunk_index": doc.get("chunk_index", 0),
                    }
                })
            # Upsert in batches of 100
            for i in range(0, len(vectors), 100):
                batch = vectors[i:i+100]
                index.upsert(vectors=batch)
        except Exception as e:
            print(f"Error adding documents: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def query(self, collection: str, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        collection = self._sanitize_name(collection)
        try:
            index = self.pc.Index(collection)
            results = index.query(vector=embedding, top_k=top_k, include_metadata=True)

            output = []
            for match in results.get("matches", []):
                metadata = match.get("metadata", {})
                output.append({
                    "content": metadata.get("content", ""),
                    "source": metadata.get("source", "unknown"),
                    "score": match.get("score", 0),
                })
            return output
        except Exception as e:
            if "not found" in str(e).lower():
                print(f"Collection '{collection}' not found", file=sys.stderr)
                sys.exit(EXIT_COLLECTION_NOT_FOUND)
            print(f"Error querying index: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def get_stats(self, collection: str) -> Dict[str, Any]:
        collection = self._sanitize_name(collection)
        try:
            index = self.pc.Index(collection)
            stats = index.describe_index_stats()
            return {
                "name": collection,
                "document_count": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", "unknown"),
                "backend": "pinecone",
            }
        except Exception as e:
            if "not found" in str(e).lower():
                print(f"Collection '{collection}' not found", file=sys.stderr)
                sys.exit(EXIT_COLLECTION_NOT_FOUND)
            print(f"Error getting stats: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)


class WeaviateBackend(VectorBackend):
    """Weaviate vector database backend."""

    def __init__(self):
        url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")

        try:
            import weaviate
            self.client = weaviate.connect_to_local(host=url.replace("http://", "").replace("https://", "").split(":")[0])
        except ImportError:
            print("Error: weaviate-client not installed. Run: pip install weaviate-client", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)
        except Exception as e:
            print(f"Error connecting to Weaviate: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def _sanitize_name(self, name: str) -> str:
        """Weaviate class names must start with uppercase letter."""
        name = name.replace("-", "_").replace(" ", "_")
        return name[0].upper() + name[1:] if name else name

    def create_collection(self, name: str, dimension: int) -> None:
        name = self._sanitize_name(name)
        try:
            if not self.client.collections.exists(name):
                self.client.collections.create(
                    name=name,
                    properties=[
                        {"name": "content", "data_type": ["text"]},
                        {"name": "source", "data_type": ["text"]},
                        {"name": "chunk_index", "data_type": ["int"]},
                    ]
                )
        except Exception as e:
            print(f"Error creating collection: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def delete_collection(self, name: str) -> None:
        name = self._sanitize_name(name)
        try:
            self.client.collections.delete(name)
        except Exception as e:
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                print(f"Collection '{name}' not found", file=sys.stderr)
                sys.exit(EXIT_COLLECTION_NOT_FOUND)
            print(f"Error deleting collection: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def list_collections(self) -> List[str]:
        try:
            collections = self.client.collections.list_all()
            return list(collections.keys())
        except Exception as e:
            print(f"Error listing collections: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def collection_exists(self, name: str) -> bool:
        name = self._sanitize_name(name)
        return self.client.collections.exists(name)

    def add_documents(self, collection: str, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        collection = self._sanitize_name(collection)
        try:
            coll = self.client.collections.get(collection)
            with coll.batch.dynamic() as batch:
                for doc, emb in zip(documents, embeddings):
                    batch.add_object(
                        properties={
                            "content": doc["content"],
                            "source": doc["source"],
                            "chunk_index": doc.get("chunk_index", 0),
                        },
                        vector=emb,
                    )
        except Exception as e:
            print(f"Error adding documents: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def query(self, collection: str, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        collection = self._sanitize_name(collection)
        try:
            coll = self.client.collections.get(collection)
            results = coll.query.near_vector(near_vector=embedding, limit=top_k, return_metadata=["distance"])

            output = []
            for obj in results.objects:
                output.append({
                    "content": obj.properties.get("content", ""),
                    "source": obj.properties.get("source", "unknown"),
                    "score": 1 - (obj.metadata.distance or 0),  # Convert distance to similarity
                })
            return output
        except Exception as e:
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                print(f"Collection '{collection}' not found", file=sys.stderr)
                sys.exit(EXIT_COLLECTION_NOT_FOUND)
            print(f"Error querying collection: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def get_stats(self, collection: str) -> Dict[str, Any]:
        collection = self._sanitize_name(collection)
        try:
            coll = self.client.collections.get(collection)
            count = coll.aggregate.over_all(total_count=True).total_count
            return {
                "name": collection,
                "document_count": count or 0,
                "backend": "weaviate",
            }
        except Exception as e:
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                print(f"Collection '{collection}' not found", file=sys.stderr)
                sys.exit(EXIT_COLLECTION_NOT_FOUND)
            print(f"Error getting stats: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)


class QdrantBackend(VectorBackend):
    """Qdrant vector database backend."""

    def __init__(self):
        url = os.environ.get("QDRANT_URL")

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            self.Distance = Distance
            self.VectorParams = VectorParams
            self.PointStruct = PointStruct

            if url:
                # Parse URL for host and port
                url = url.replace("http://", "").replace("https://", "")
                parts = url.split(":")
                host = parts[0]
                port = int(parts[1]) if len(parts) > 1 else 6333
                self.client = QdrantClient(host=host, port=port)
            else:
                # Use local persistent storage
                storage_path = get_storage_dir() / "qdrant"
                storage_path.mkdir(parents=True, exist_ok=True)
                self.client = QdrantClient(path=str(storage_path))
        except ImportError:
            print("Error: qdrant-client not installed. Run: pip install qdrant-client", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def create_collection(self, name: str, dimension: int) -> None:
        try:
            collections = self.client.get_collections().collections
            if name not in [c.name for c in collections]:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=self.VectorParams(size=dimension, distance=self.Distance.COSINE)
                )
        except Exception as e:
            print(f"Error creating collection: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def delete_collection(self, name: str) -> None:
        try:
            self.client.delete_collection(collection_name=name)
        except Exception as e:
            if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                print(f"Collection '{name}' not found", file=sys.stderr)
                sys.exit(EXIT_COLLECTION_NOT_FOUND)
            print(f"Error deleting collection: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def list_collections(self) -> List[str]:
        try:
            collections = self.client.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            print(f"Error listing collections: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def collection_exists(self, name: str) -> bool:
        return name in self.list_collections()

    def add_documents(self, collection: str, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        try:
            points = []
            for i, (doc, emb) in enumerate(zip(documents, embeddings)):
                # Use hash of doc id as integer id for Qdrant
                int_id = int(hashlib.md5(doc["id"].encode()).hexdigest()[:15], 16)
                points.append(self.PointStruct(
                    id=int_id,
                    vector=emb,
                    payload={
                        "content": doc["content"],
                        "source": doc["source"],
                        "chunk_index": doc.get("chunk_index", 0),
                        "doc_id": doc["id"],
                    }
                ))
            self.client.upsert(collection_name=collection, points=points)
        except Exception as e:
            print(f"Error adding documents: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def query(self, collection: str, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        try:
            results = self.client.search(
                collection_name=collection,
                query_vector=embedding,
                limit=top_k,
            )

            output = []
            for hit in results:
                payload = hit.payload or {}
                output.append({
                    "content": payload.get("content", ""),
                    "source": payload.get("source", "unknown"),
                    "score": hit.score,
                })
            return output
        except Exception as e:
            if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                print(f"Collection '{collection}' not found", file=sys.stderr)
                sys.exit(EXIT_COLLECTION_NOT_FOUND)
            print(f"Error querying collection: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)

    def get_stats(self, collection: str) -> Dict[str, Any]:
        try:
            info = self.client.get_collection(collection_name=collection)
            return {
                "name": collection,
                "document_count": info.points_count or 0,
                "dimension": info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else "unknown",
                "backend": "qdrant",
            }
        except Exception as e:
            if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                print(f"Collection '{collection}' not found", file=sys.stderr)
                sys.exit(EXIT_COLLECTION_NOT_FOUND)
            print(f"Error getting stats: {e}", file=sys.stderr)
            sys.exit(EXIT_BACKEND_ERROR)


def get_backend(backend_name: str) -> VectorBackend:
    """Factory function to get the appropriate backend."""
    backends = {
        "chromadb": ChromaDBBackend,
        "pinecone": PineconeBackend,
        "weaviate": WeaviateBackend,
        "qdrant": QdrantBackend,
    }

    if backend_name not in backends:
        print(f"Error: Unknown backend '{backend_name}'. Choose from: {', '.join(backends.keys())}", file=sys.stderr)
        sys.exit(EXIT_INVALID_ARGS)

    return backends[backend_name]()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind(". ")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)
            if break_point > chunk_size // 2:
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


def read_file(filepath: str) -> Tuple[str, str]:
    """Read file content and return (content, source)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return content, filepath
    except UnicodeDecodeError:
        print(f"Warning: Skipping binary file: {filepath}", file=sys.stderr)
        return "", filepath
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
        return "", filepath


def cmd_index(args) -> int:
    """Index documents into a collection."""
    if not args.collection:
        print("Error: --collection is required", file=sys.stderr)
        return EXIT_INVALID_ARGS

    # Expand glob patterns
    files = []
    for pattern in args.files:
        expanded = glob.glob(pattern, recursive=True)
        if not expanded:
            # Treat as literal file path
            if os.path.isfile(pattern):
                files.append(pattern)
            else:
                print(f"Warning: No files found matching: {pattern}", file=sys.stderr)
        else:
            files.extend(expanded)

    if not files:
        print("Error: No files to index", file=sys.stderr)
        return EXIT_INVALID_ARGS

    # Remove duplicates while preserving order
    files = list(dict.fromkeys(files))

    print(f"Indexing {len(files)} file(s) into collection '{args.collection}'...")

    # Initialize embedding model and backend
    model = EmbeddingModel(args.model)
    backend = get_backend(args.backend)

    # Create collection
    backend.create_collection(args.collection, model.dimension)

    # Process files
    all_documents = []
    all_texts = []

    for filepath in files:
        content, source = read_file(filepath)
        if not content:
            continue

        chunks = chunk_text(content, args.chunk_size, args.overlap)
        for i, chunk in enumerate(chunks):
            doc_id = generate_doc_id(chunk, source + f":{i}")
            all_documents.append({
                "id": doc_id,
                "content": chunk,
                "source": source,
                "chunk_index": i,
            })
            all_texts.append(chunk)

    if not all_documents:
        print("Error: No content to index", file=sys.stderr)
        return EXIT_INVALID_ARGS

    # Generate embeddings in batches
    print(f"Generating embeddings for {len(all_documents)} chunk(s)...")
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i+batch_size]
        embeddings = model.encode(batch)
        all_embeddings.extend(embeddings)
        print(f"  Processed {min(i+batch_size, len(all_texts))}/{len(all_texts)} chunks", end="\r")

    print()  # New line after progress

    # Add to backend
    print("Storing in vector database...")
    backend.add_documents(args.collection, all_documents, all_embeddings)

    print(f"Successfully indexed {len(all_documents)} chunk(s) from {len(files)} file(s)")
    return EXIT_SUCCESS


def cmd_query(args) -> int:
    """Query for similar documents."""
    if not args.collection:
        print("Error: --collection is required", file=sys.stderr)
        return EXIT_INVALID_ARGS

    if not args.query_text:
        print("Error: Query text is required", file=sys.stderr)
        return EXIT_INVALID_ARGS

    # Initialize embedding model and backend
    model = EmbeddingModel(args.model)
    backend = get_backend(args.backend)

    # Check collection exists
    if not backend.collection_exists(args.collection):
        print(f"Collection '{args.collection}' not found", file=sys.stderr)
        return EXIT_COLLECTION_NOT_FOUND

    # Generate query embedding
    query_embedding = model.encode([args.query_text])[0]

    # Query
    results = backend.query(args.collection, query_embedding, args.top_k)

    if not results:
        print("No results found")
        return EXIT_SUCCESS

    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nTop {len(results)} results for: \"{args.query_text}\"\n")
        print("-" * 80)
        for i, result in enumerate(results, 1):
            score = result.get("score", 0)
            source = result.get("source", "unknown")
            content = result.get("content", "")

            # Truncate content for display
            if len(content) > 300:
                content = content[:300] + "..."

            print(f"\n[{i}] Score: {score:.4f}")
            print(f"    Source: {source}")
            print(f"    Content: {content}")
        print("\n" + "-" * 80)

    return EXIT_SUCCESS


def cmd_list_collections(args) -> int:
    """List all collections."""
    backend = get_backend(args.backend)
    collections = backend.list_collections()

    if not collections:
        print("No collections found")
        return EXIT_SUCCESS

    if args.json:
        print(json.dumps(collections, indent=2))
    else:
        print(f"Collections ({len(collections)}):")
        for name in sorted(collections):
            print(f"  - {name}")

    return EXIT_SUCCESS


def cmd_stats(args) -> int:
    """Show collection statistics."""
    if not args.collection:
        print("Error: --collection is required", file=sys.stderr)
        return EXIT_INVALID_ARGS

    backend = get_backend(args.backend)
    stats = backend.get_stats(args.collection)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(f"Collection: {stats['name']}")
        print(f"  Backend: {stats.get('backend', 'unknown')}")
        print(f"  Documents: {stats.get('document_count', 'unknown')}")
        print(f"  Dimension: {stats.get('dimension', 'unknown')}")

    return EXIT_SUCCESS


def cmd_delete(args) -> int:
    """Delete a collection."""
    if not args.collection:
        print("Error: --collection is required", file=sys.stderr)
        return EXIT_INVALID_ARGS

    backend = get_backend(args.backend)

    if not args.force:
        response = input(f"Are you sure you want to delete collection '{args.collection}'? [y/N] ")
        if response.lower() != "y":
            print("Cancelled")
            return EXIT_SUCCESS

    backend.delete_collection(args.collection)
    print(f"Collection '{args.collection}' deleted")
    return EXIT_SUCCESS


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with comprehensive help."""

    # Custom formatter that preserves newlines in description
    class RawDescriptionDefaultsHelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        prog="vector_search",
        description=textwrap.dedent("""
            Semantic document search CLI using vector embeddings.

            This tool indexes documents and enables semantic similarity search
            using various vector database backends and sentence-transformers
            for automatic embedding generation.

            SUPPORTED BACKENDS:
              chromadb  - Local persistent storage (default, no server required)
              pinecone  - Cloud-based vector database (requires PINECONE_API_KEY)
              weaviate  - Self-hosted or cloud (uses WEAVIATE_URL)
              qdrant    - Local or cloud deployment (uses QDRANT_URL)

            ENVIRONMENT VARIABLES:
              VECTOR_SEARCH_DIR   Storage directory (default: ~/.vector_search/)
              PINECONE_API_KEY    API key for Pinecone backend
              WEAVIATE_URL        Weaviate server URL (default: http://localhost:8080)
              QDRANT_URL          Qdrant server URL (optional, uses local storage if not set)

            EXIT CODES:
              0 - Success
              1 - Collection not found
              2 - Backend error (connection, API, etc.)
              3 - Invalid arguments
        """),
        formatter_class=RawDescriptionDefaultsHelpFormatter,
        epilog=textwrap.dedent("""
            EXAMPLES:
              # Index markdown files into a collection
              %(prog)s index "docs/*.md" --collection knowledge-base

              # Index a single file
              %(prog)s index README.md --collection my-docs

              # Index with a specific backend
              %(prog)s index "*.txt" --collection notes --backend qdrant

              # Query for similar documents
              %(prog)s query "How do I configure authentication?" --collection knowledge-base

              # Query with more results
              %(prog)s query "deployment" --collection my-docs --top-k 10

              # Get JSON output for programmatic use
              %(prog)s query "setup instructions" --collection docs --json

              # List all collections
              %(prog)s list-collections

              # Get collection statistics
              %(prog)s stats --collection knowledge-base

              # Delete a collection
              %(prog)s delete --collection old-docs

              # Delete without confirmation
              %(prog)s delete --collection temp --force

            For more information, visit: https://github.com/example/vector_search
        """)
    )

    # Global options
    parser.add_argument(
        "--backend", "-b",
        choices=["chromadb", "pinecone", "weaviate", "qdrant"],
        default="chromadb",
        help="Vector database backend to use"
    )
    parser.add_argument(
        "--model", "-m",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model for embeddings"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser(
        "index",
        help="Index documents for semantic search",
        description=textwrap.dedent("""
            Index documents into a collection for semantic search.

            Supports glob patterns (e.g., "docs/*.md", "**/*.txt") and
            multiple file paths. Documents are automatically chunked
            and embedded using sentence-transformers.
        """),
        formatter_class=RawDescriptionDefaultsHelpFormatter,
        epilog=textwrap.dedent("""
            EXAMPLES:
              %(prog)s "docs/*.md" --collection knowledge-base
              %(prog)s README.md CHANGELOG.md --collection my-docs
              %(prog)s "**/*.py" --collection code --chunk-size 500
        """)
    )
    index_parser.add_argument(
        "files",
        nargs="+",
        help="File paths or glob patterns to index"
    )
    index_parser.add_argument(
        "--collection", "-c",
        required=True,
        help="Collection name to index into"
    )
    index_parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum characters per chunk"
    )
    index_parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Character overlap between chunks"
    )
    index_parser.set_defaults(func=cmd_index)

    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Search for similar documents",
        description=textwrap.dedent("""
            Query a collection for semantically similar documents.

            The query text is embedded and compared against all
            documents in the collection using cosine similarity.
        """),
        formatter_class=RawDescriptionDefaultsHelpFormatter,
        epilog=textwrap.dedent("""
            EXAMPLES:
              %(prog)s "How do I configure authentication?" --collection docs
              %(prog)s "deployment steps" --collection knowledge-base --top-k 10
              %(prog)s "error handling" --collection code --json
        """)
    )
    query_parser.add_argument(
        "query_text",
        help="Query text for similarity search"
    )
    query_parser.add_argument(
        "--collection", "-c",
        required=True,
        help="Collection name to search"
    )
    query_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results to return"
    )
    query_parser.set_defaults(func=cmd_query)

    # List collections command
    list_parser = subparsers.add_parser(
        "list-collections",
        help="List all collections",
        description="List all collections in the vector database.",
        formatter_class=RawDescriptionDefaultsHelpFormatter
    )
    list_parser.set_defaults(func=cmd_list_collections)

    # Stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show collection statistics",
        description="Display statistics for a collection including document count and dimensions.",
        formatter_class=RawDescriptionDefaultsHelpFormatter
    )
    stats_parser.add_argument(
        "--collection", "-c",
        required=True,
        help="Collection name"
    )
    stats_parser.set_defaults(func=cmd_stats)

    # Delete command
    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete a collection",
        description="Delete a collection and all its documents.",
        formatter_class=RawDescriptionDefaultsHelpFormatter
    )
    delete_parser.add_argument(
        "--collection", "-c",
        required=True,
        help="Collection name to delete"
    )
    delete_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Delete without confirmation"
    )
    delete_parser.set_defaults(func=cmd_delete)

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return EXIT_SUCCESS

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
