from db.session import get_session
from db.models import QueryLog, LowRecallEvent

def delete_query_by_id():
    query_id_str = input("Enter the ID of the QueryLog you want to delete: ").strip()
    if not query_id_str.isdigit():
        print("❌ Invalid ID. Please enter a valid number.")
        return
        
    query_id = int(query_id_str)
    session = get_session()
    try:
        # Find the specific query log by ID
        log = session.query(QueryLog).filter(QueryLog.id == query_id).first()
        
        if not log:
            print(f"No query found with ID: {query_id}")
            return
            
        print(f"Found Query [ID: {log.id}]: '{log.query[:50]}...'")
        
        # Find and delete any associated LowRecallEvents first to avoid orphaned records
        events = session.query(LowRecallEvent).filter(LowRecallEvent.query_log_id == log.id).all()
        for event in events:
            session.delete(event)
            
        # Delete the query log itself
        session.delete(log)
        
        # Commit the transaction
        session.commit()
        print(f"✅ Successfully deleted QueryLog ID {query_id}")
        if events:
            print(f"✅ Also deleted {len(events)} associated LowRecallEvent(s).")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error deleting query: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    delete_query_by_id()
