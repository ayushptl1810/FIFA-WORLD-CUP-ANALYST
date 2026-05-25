import os
from dotenv import load_dotenv
import redis
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

load_dotenv()

def check(name, fn):
    try:
        fn()
        print(f"  OK   {name}")
    except Exception as e:
        print(f"  FAIL {name} — {e}")

print("\nTactIQ — Stage 1 Docker Connection Check\n" + "─"*40)

# # API-Football
# def test_apifootball():
#     import requests
#     r = requests.get(
#         "https://api-football-v1.p.rapidapi.com/v3/status",
#         headers={"X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY"),
#                  "X-RapidAPI-Host": os.getenv("RAPIDAPI_HOST")}
#     )
#     assert r.status_code == 200
# check("API-Football (status endpoint)", test_apifootball)

# Redis
def test_redis():
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    r = redis.from_url(redis_url)
    assert r.ping()
check("Redis (ping)", test_redis)

# Neo4j
def test_neo4j():
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "tactiq1234")
    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_user, neo4j_password)
    )
    with driver.session() as s:
        s.run("RETURN 1")
    driver.close()
check("Neo4j (bolt connection)", test_neo4j)

# Qdrant
def test_qdrant():
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)
    client.get_collections()
check("Qdrant (list collections)", test_qdrant)

print("─"*40)
print("Run `docker compose up -d` first if Redis/Neo4j/Qdrant fail.\n")