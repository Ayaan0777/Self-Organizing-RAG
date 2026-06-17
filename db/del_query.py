from db.session import get_session
from db.models import (
    QueryLog, LowRecallEvent,
    RepairReport, AdaptationLog, ChunkSnapshot,
)


def delete_query_by_id():
    query_id_str = input("Enter the ID of the QueryLog you want to delete: ").strip()
    if not query_id_str.isdigit():
        print("❌ Invalid ID. Please enter a valid number.")
        return

    query_id = int(query_id_str)
    session = get_session()
    try:
        log = session.query(QueryLog).filter(QueryLog.id == query_id).first()
        if not log:
            print(f"No query found with ID: {query_id}")
            return

        print(f"Found Query [ID: {log.id}]: '{log.query[:50]}...'")

        # Cascade-delete child rows in FK order to avoid orphans.
        # SQLite doesn't enforce FK constraints by default, so we have to do
        # this manually: child rows of LowRecallEvent first, then events, then log.
        events = session.query(LowRecallEvent).filter(
            LowRecallEvent.query_log_id == log.id
        ).all()

        reports_deleted = 0
        adaptations_deleted = 0
        snapshots_deleted = 0
        for event in events:
            reports_deleted += session.query(RepairReport).filter(
                RepairReport.event_id == event.id
            ).delete()
            adaptations_deleted += session.query(AdaptationLog).filter(
                AdaptationLog.event_id == event.id
            ).delete()
            snapshots_deleted += session.query(ChunkSnapshot).filter(
                ChunkSnapshot.event_id == event.id
            ).delete()
            session.delete(event)

        session.delete(log)
        session.commit()

        print(f"✅ Deleted QueryLog ID {query_id}")
        if events:
            print(f"   ↳ {len(events)} LowRecallEvent(s)")
            if reports_deleted:
                print(f"   ↳ {reports_deleted} RepairReport(s)")
            if adaptations_deleted:
                print(f"   ↳ {adaptations_deleted} AdaptationLog(s)")
            if snapshots_deleted:
                print(f"   ↳ {snapshots_deleted} ChunkSnapshot(s)")

    except Exception as e:
        session.rollback()
        print(f"❌ Error deleting query: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    delete_query_by_id()
