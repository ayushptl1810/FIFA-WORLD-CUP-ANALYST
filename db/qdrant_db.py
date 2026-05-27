import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
load_dotenv()

def init_qdrant_schema():
    print("\n" + "="*60)
    print("INITIALIZING QDRANT VECTOR COLLECTIONS")
    print("="*60)

    # 1. Fetch connection details from environment variables
    qdrant_url = os.getenv("QDRANT_URL") or "http://localhost:6333"
    grpc_port = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    print(f"[Auth] Connecting to Qdrant at: {qdrant_url}")
    
    # Initialize the Qdrant Client (setting check_compatibility=False avoids version mismatch logs)
    client = QdrantClient(url=qdrant_url, check_compatibility=False, prefer_grpc=True, grpc_port=grpc_port)

    # 2. Check if the 'historical_states' collection already exists
    existing = [c.name for c in client.get_collections().collections]

    collection_name = "historical_states"
    if collection_name in existing:
        print(f"\n[1] Dropping existing collection to resize: {collection_name}...")
        client.delete_collection(collection_name=collection_name)
        existing.remove(collection_name)

    print(f"\n[1] Creating collection: {collection_name} with vector size = 24...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=24, distance=Distance.COSINE),
    )
    print(f"    ✅ Collection created: {collection_name}")

    # 3. Verify collection parameters robustly (handling object vs dict representation)
    print("\n" + "─"*40)
    print("VERIFYING COLLECTION SCHEMA")
    print("─"*40)
    
    info = client.get_collection(collection_name)
    vectors = info.config.params.vectors

    vector_size = None
    vector_distance = None

    if vectors is not None:
        if isinstance(vectors, dict):
            # If Qdrant returns a dictionary of named vectors
            first_key = list(vectors.keys())[0]
            first_vector = vectors[first_key]
            
            # Check if the values inside are objects or sub-dicts
            if hasattr(first_vector, "size"):
                vector_size = first_vector.size
                vector_distance = first_vector.distance
            else:
                vector_size = getattr(first_vector, "size", None)
                vector_distance = getattr(first_vector, "distance", None)
        else:
            # If Qdrant returns a single VectorParams object directly
            if hasattr(vectors, "size"):
                vector_size = vectors.size
                vector_distance = vectors.distance
            else:
                # Safe fallback lookup
                vector_size = getattr(vectors, "size", None)
                vector_distance = getattr(vectors, "distance", None)

    print(f"✅ Collection Name: {collection_name}")
    print(f"✅ Vector size:     {vector_size}")
    print(f"✅ Distance Metric: {vector_distance}")
    print("\n✅ Qdrant database schema initialization completed successfully!")

if __name__ == "__main__":
    init_qdrant_schema()