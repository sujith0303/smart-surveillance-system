from src.database.event_database import SurveillanceEventDB
import json

# Initialize database
db = SurveillanceEventDB('data/outputs/surveillance.db')

# Load tracking report
with open('data/outputs/tracking_report.json', 'r') as f:
    report = json.load(f)

# Insert tracks into database
print(" Inserting tracks into database...")
for track in report['tracks']:
    db.insert_track(track['track_id'], track)
    
    # Insert behaviors
    for behavior in track.get('behaviors', []):
        db.insert_behavior(track['track_id'], behavior)

print(f" Inserted {len(report['tracks'])} tracks")

# Query some data
print("\n🔍 Query Examples:")

# Get statistics
stats = db.get_statistics()
print(f"\n Statistics:")
print(f"   Total unique tracks: {stats['total_unique_tracks']}")
print(f"   Behaviors by type: {stats.get('behaviors_by_type', {})}")

# Query behaviors
behaviors = db.query_behaviors(behavior_type='fast_movement')
print(f"\n⚡ Fast movement events: {len(behaviors[:10])} (showing first 10)")

# Query by track ID
track_history = db.query_track_history(1)
print(f"\n Track ID 1 history:")
print(f"   Duration: {track_history['track'][3]:.1f}s")
print(f"   Total behaviors: {len(track_history['behaviors'])}")

db.close()
print("\n Database test complete!")
