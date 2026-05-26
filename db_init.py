import sys
from dotenv import load_dotenv
load_dotenv()

from db.redis_cache import init_redis_streams
from db.neo4j_db import init_neo4j_schema
from db.qdrant_db import init_qdrant_schema

def run_master_init():
    print("\n" + "="*80)
    print("                TACTIQ - MASTER DATABASE INITIALIZATION SYSTEM                ")
    print("="*80)
    print("Starting sequential database initialization...\n")
    
    status = {
        "Redis Streams": False,
        "Neo4j Schema": False,
        "Qdrant Collections": False
    }

    # 1. Initialize Redis
    try:
        init_redis_streams()
        status["Redis Streams"] = True
    except Exception as e:
        print(f"\n❌ Redis streams initialization failed: {e}")

    # 2. Initialize Neo4j
    try:
        init_neo4j_schema()
        status["Neo4j Schema"] = True
    except Exception as e:
        print(f"\n❌ Neo4j database schema initialization failed: {e}")

    # 3. Initialize Qdrant
    try:
        init_qdrant_schema()
        status["Qdrant Collections"] = True
    except Exception as e:
        print(f"\n❌ Qdrant vector collections initialization failed: {e}")

    # 4. Final Status Summary
    print("\n" + "="*80)
    print("                       DATABASE INITIALIZATION SUMMARY                        ")
    print("="*80)
    
    all_success = True
    for db_name, success in status.items():
        icon = "✅ SUCCESS" if success else "❌ FAILED "
        print(f"   • {db_name:<20} : {icon}")
        if not success:
            all_success = False

    print("─"*80)
    if all_success:
        print("🎉 Congratulations! All databases have been successfully initialized and are ready!")
    else:
        print("⚠️ Some database initializations failed. Please inspect the logs above, check your")
        print("   Docker containers ('docker compose ps'), and run 'python db_init.py' again.")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_master_init()
