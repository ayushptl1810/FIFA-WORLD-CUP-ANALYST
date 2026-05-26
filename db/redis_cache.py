import os
from dotenv import load_dotenv
import redis
load_dotenv()

def get_redis_client():
    redis_url = os.getenv("REDIS_URL") or "redis://localhost:6379"
    return redis.from_url(redis_url)

def init_redis_streams():
    print("\n" + "="*60)
    print("INITIALIZING REDIS STREAMS & CONSUMER GROUPS")
    print("="*60)

    # 1. Fetch connection details
    r = get_redis_client()
    print(f"[Auth] Connecting to Redis at: {os.getenv('REDIS_URL', 'redis://localhost:6379')}")

    streams = ["match:events", "match:positions"]

    # 2. Idempotently create streams and consumer groups
    print("\n[1] Defining Redis Streams & Consumer Groups...")
    for stream in streams:
        try:
            # MKSTREAM creates the stream if it doesn't exist
            r.xgroup_create(
                name=stream,
                groupname="graph_updater",
                id="$",
                mkstream=True
            )
            print(f"    ✅ Created stream + consumer group: {stream} (group: graph_updater)")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                print(f"    ✅ Consumer group already exists:  {stream} (group: graph_updater)")
            else:
                print(f"    ❌ Error creating group for {stream}: {e}")
                raise

    # 3. Verification
    print("\n" + "─"*40)
    print("VERIFYING STREAMS CREATION")
    print("─"*40)
    
    for stream in streams:
        try:
            info = r.xinfo_stream(stream)
            groups = r.xinfo_groups(stream)
            
            # Resolve static type checker ambiguity between sync/async Redis return types
            info_dict = info if isinstance(info, dict) else {}
            groups_list = groups if isinstance(groups, list) else []
            
            # Safely extract group names to print
            group_names = [g.get("name") if isinstance(g, dict) else getattr(g, "name", str(g)) for g in groups_list]
            
            print(f"✅ Stream Name: {stream:<16} | Length: {info_dict.get('length'):<3} | Groups: {group_names}")
        except Exception as e:
            print(f"❌ Failed to verify stream {stream}: {e}")

    print("\n✅ Redis database schema initialization completed successfully!")

if __name__ == "__main__":
    init_redis_streams()