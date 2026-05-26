import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
load_dotenv()

def get_neo4j_driver():
    uri = os.getenv("NEO4J_URI") or "bolt://localhost:7687"
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    
    return GraphDatabase.driver(uri, auth=(user, password))

def init_neo4j_schema():
    print("\n" + "="*60)
    print("INITIALIZING NEO4J DATABASE SCHEMA")
    print("="*60)
    
    # 1. Uniqueness Constraints
    constraints = [
        "CREATE CONSTRAINT match_id_unique IF NOT EXISTS FOR (m:Match) REQUIRE m.match_id IS UNIQUE",
        "CREATE CONSTRAINT player_id_unique IF NOT EXISTS FOR (p:Player) REQUIRE p.player_id IS UNIQUE",
        "CREATE CONSTRAINT team_id_unique IF NOT EXISTS FOR (t:Team) REQUIRE t.team_id IS UNIQUE",
        "CREATE CONSTRAINT zone_id_unique IF NOT EXISTS FOR (z:Zone) REQUIRE z.zone_id IS UNIQUE",
        "CREATE CONSTRAINT gamestate_id_unique IF NOT EXISTS FOR (g:GameState) REQUIRE g.state_id IS UNIQUE"
    ]
    
    # 2. Lookup Indexes
    indexes = [
        "CREATE INDEX player_team_idx IF NOT EXISTS FOR (p:Player) ON (p.team_id)",
        "CREATE INDEX gamestate_minute_idx IF NOT EXISTS FOR (g:GameState) ON (g.minute)",
        "CREATE INDEX match_competition_idx IF NOT EXISTS FOR (m:Match) ON (m.competition_id)"
    ]
    
    driver = get_neo4j_driver()
    
    with driver.session() as session:
        # Run Uniqueness Constraints
        print("\n[1] Defining Uniqueness Constraints...")
        for q in constraints:
            try:
                session.run(q)
                constraint_name = q.split("IF NOT EXISTS")[0].split("CREATE CONSTRAINT")[1].strip()
                print(f"    ✅ Constraint ensured: {constraint_name}")
            except Exception as e:
                print(f"    ❌ Error creating constraint: {e}")
                
        # Run Lookup Indexes
        print("\n[2] Defining Lookup Indexes...")
        for q in indexes:
            try:
                session.run(q)
                index_name = q.split("IF NOT EXISTS")[0].split("CREATE INDEX")[1].strip()
                print(f"    ✅ Index ensured: {index_name}")
            except Exception as e:
                print(f"    ❌ Error creating index: {e}")
                
        # 3. Verification
        print("\n" + "─"*40)
        print("VERIFYING SCHEMA CREATION")
        print("─"*40)
        
        # Verify Constraints
        print("\nActive Uniqueness Constraints (SHOW CONSTRAINTS):")
        res_constraints = session.run("SHOW CONSTRAINTS")
        constraints_list = []
        for rec in res_constraints:
            constraints_list.append({
                "Name": rec.get("name"),
                "Type": rec.get("type"),
                "Entity": rec.get("entityType"),
                "Labels": rec.get("labelsOrTypes"),
                "Properties": rec.get("properties"),
                "State": rec.get("state")
            })
        
        if constraints_list:
            df_c = pd.DataFrame(constraints_list)
            print(df_c.to_string(index=False))
        else:
            print("    No active uniqueness constraints found.")
            
        # Verify Indexes
        print("\nActive Lookup Indexes (SHOW INDEXES):")
        res_indexes = session.run("SHOW INDEXES")
        indexes_list = []
        for rec in res_indexes:
            indexes_list.append({
                "Name": rec.get("name"),
                "Type": rec.get("type"),
                "State": rec.get("state"),
                "Labels": rec.get("labelsOrTypes"),
                "Properties": rec.get("properties")
            })
            
        if indexes_list:
            df_i = pd.DataFrame(indexes_list)
            print(df_i.to_string(index=False))
        else:
            print("    No active lookup indexes found.")
            
    driver.close()
    print("\n✅ Neo4j database schema initialization completed successfully!")

if __name__ == "__main__":
    init_neo4j_schema()
