# Comprehensive System Test & Validation
# File: test_complete_system.py

import json
import os
import sys
import sqlite3
from datetime import datetime

# Fix imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src', 'langchain'))

from vector_store import SurveillanceVectorStore
from query_engine import SurveillanceQueryEngine

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_detection_results():
    """Test Object Detection Pipeline"""
    print_section("📹 TESTING: OBJECT DETECTION")
    
    detection_file = 'data/outputs/surveillance_detections.json'
    
    if not os.path.exists(detection_file):
        print(" Detection results not found!")
        return False
    
    with open(detection_file, 'r') as f:
        detections = json.load(f)
    
    print(f"Detection file loaded: {len(detections)} events")
    
    # Analyze detections
    object_types = {}
    colors = {}
    
    for det in detections:
        obj_type = det.get('object_type', 'unknown')
        color = det.get('attributes', {}).get('color', 'unknown')
        
        object_types[obj_type] = object_types.get(obj_type, 0) + 1
        colors[color] = colors.get(color, 0) + 1
    
    print(f"\n Object Type Distribution:")
    for obj_type, count in sorted(object_types.items(), key=lambda x: -x[1]):
        print(f"   {obj_type:15s}: {count:4d} ({count/len(detections)*100:.1f}%)")
    
    print(f"\n Color Distribution:")
    for color, count in sorted(colors.items(), key=lambda x: -x[1])[:10]:
        print(f"   {color:15s}: {count:4d} ({count/len(detections)*100:.1f}%)")
    
    return True

def test_tracking_results():
    """Test Multi-Object Tracking Pipeline"""
    print_section(" TESTING: MULTI-OBJECT TRACKING")
    
    tracking_file = 'data/outputs/tracking_report.json'
    
    if not os.path.exists(tracking_file):
        print(" Tracking results not found!")
        return False
    
    with open(tracking_file, 'r') as f:
        report = json.load(f)
    
    summary = report.get('summary', {})
    tracks = report.get('tracks', [])
    
    print(f" Tracking report loaded")
    print(f"\n Summary Statistics:")
    print(f"   Total unique tracks: {summary.get('total_tracks', 0)}")
    print(f"   Suspicious tracks: {summary.get('suspicious_tracks', 0)}")
    print(f"   Total behaviors: {summary.get('total_behaviors', 0)}")
    
    # Analyze track durations
    durations = [t['duration'] for t in tracks]
    avg_duration = sum(durations) / len(durations) if durations else 0
    max_duration = max(durations) if durations else 0
    min_duration = min(durations) if durations else 0
    
    print(f"\n  Track Duration Analysis:")
    print(f"   Average: {avg_duration:.2f} seconds")
    print(f"   Maximum: {max_duration:.2f} seconds")
    print(f"   Minimum: {min_duration:.2f} seconds")
    
    # Behavior analysis
    behavior_counts = {}
    for track in tracks:
        for behavior in track.get('behaviors', []):
            b_type = behavior['type']
            behavior_counts[b_type] = behavior_counts.get(b_type, 0) + 1
    
    print(f"\n  Behavior Detection:")
    for behavior, count in sorted(behavior_counts.items(), key=lambda x: -x[1]):
        print(f"   {behavior:20s}: {count:4d} events")
    
    return True

def test_database():
    """Test Event Database"""
    print_section(" TESTING: EVENT DATABASE")
    
    db_file = 'data/outputs/surveillance.db'
    
    if not os.path.exists(db_file):
        print(" Database not found!")
        return False
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f" Database connected")
    print(f"\n Tables: {', '.join([t[0] for t in tables])}")
    
    # Count records
    cursor.execute("SELECT COUNT(*) FROM tracks")
    track_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM behaviors")
    behavior_count = cursor.fetchone()[0]
    
    print(f"\n Record Counts:")
    print(f"   Tracks: {track_count}")
    print(f"   Behaviors: {behavior_count}")
    
    # Check for any data quality issues
    cursor.execute("SELECT COUNT(*) FROM tracks WHERE duration < 0")
    invalid_tracks = cursor.fetchone()[0]
    
    if invalid_tracks > 0:
        print(f"\n  Warning: {invalid_tracks} tracks with negative duration")
    else:
        print(f"\n Data quality check passed")
    
    conn.close()
    return True

def test_vector_store():
    """Test Vector Database"""
    print_section(" TESTING: VECTOR DATABASE")
    
    chroma_dir = 'data/outputs/chroma_db'
    
    if not os.path.exists(chroma_dir):
        print(" Vector database not found!")
        return False
    
    try:
        vector_store = SurveillanceVectorStore(chroma_dir)
        stats = vector_store.get_statistics()
        
        print(f" Vector store loaded")
        print(f"\n Statistics:")
        print(f"   Total indexed items: {stats['total_indexed']}")
        
        # Test basic search
        print(f"\n Testing basic search...")
        results = vector_store.search("person", n_results=3)
        
        if results['documents']:
            print(f" Search working - found {len(results['documents'])} results")
            print(f"\n Sample result:")
            print(f"   {results['documents'][0][:100]}...")
        else:
            print("  Search returned no results")
            return False
        
        return True
        
    except Exception as e:
        print(f" Vector store error: {str(e)}")
        return False

def test_query_engine():
    """Test Natural Language Query Engine"""
    print_section(" TESTING: QUERY ENGINE")
    
    try:
        print(" Loading query engine...")
        vector_store = SurveillanceVectorStore('data/outputs/chroma_db')
        query_engine = SurveillanceQueryEngine(vector_store)
        
        print(" Query engine loaded")
        
        # Test queries
        test_queries = [
            "Show me people",
            "Find red objects",
            "Show vehicles",
        ]
        
        print(f"\n Testing {len(test_queries)} queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Test {i}: '{query}'")
            try:
                response = query_engine.query(query, max_results=3)
                if response:
                    print(f"   Response generated ({len(response)} chars)")
                else:
                    print(f"     Empty response")
            except Exception as e:
                print(f"    Error: {str(e)}")
                return False
        
        print(f"\n All query tests passed")
        return True
        
    except Exception as e:
        print(f" Query engine error: {str(e)}")
        return False

def test_file_structure():
    """Test project file structure"""
    print_section(" PROJECT STRUCTURE")
    
    required_files = {
        'Core Files': [
            'README.md',
            'requirements.txt',
            '.gitignore'
        ],
        'Detection': [
            'src/detection/detection_pipeline.py'
        ],
        'Tracking': [
            'src/tracking/tracking_pipeline.py'
        ],
        'Database': [
            'src/database/event_database.py'
        ],
        'LangChain': [
            'src/langchain/vector_store.py',
            'src/langchain/query_engine.py'
        ],
        'API': [
            'src/api/main.py'
        ],
        'Data': [
            'data/models/yolov8x.pt',
            'data/videos/test_surveillance.mp4'
        ],
        'Output': [
            'data/outputs/surveillance_detections.json',
            'data/outputs/tracking_report.json',
            'data/outputs/surveillance.db'
        ]
    }
    
    all_good = True
    
    for category, files in required_files.items():
        print(f"\n {category}:")
        for file in files:
            exists = os.path.exists(file)
            status = "" if exists else "❌"
            print(f"   {status} {file}")
            if not exists:
                all_good = False
    
    return all_good

def generate_report():
    """Generate comprehensive system report"""
    print_section(" COMPREHENSIVE SYSTEM REPORT")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'tests_run': [],
        'tests_passed': 0,
        'tests_failed': 0
    }
    
    tests = [
        ('File Structure', test_file_structure),
        ('Object Detection', test_detection_results),
        ('Multi-Object Tracking', test_tracking_results),
        ('Event Database', test_database),
        ('Vector Store', test_vector_store),
        ('Query Engine', test_query_engine)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            status = "PASS" if result else "FAIL"
            report['tests_run'].append({'name': test_name, 'status': status})
            
            if result:
                report['tests_passed'] += 1
            else:
                report['tests_failed'] += 1
                
        except Exception as e:
            print(f"\n Test '{test_name}' crashed: {str(e)}")
            report['tests_run'].append({'name': test_name, 'status': 'ERROR'})
            report['tests_failed'] += 1
    
    # Final summary
    print_section(" FINAL SUMMARY")
    
    total_tests = len(tests)
    success_rate = (report['tests_passed'] / total_tests * 100)
    
    print(f"\n Test Results:")
    print(f"   Total tests: {total_tests}")
    print(f"   Passed: {report['tests_passed']} ")
    print(f"   Failed: {report['tests_failed']} ")
    print(f"   Success rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print(f"\n ALL TESTS PASSED! System fully operational.")
    elif success_rate >= 80:
        print(f"\n  Most tests passed. Review failures above.")
    else:
        print(f"\n Multiple failures detected. Needs attention.")
    
    # Save report
    report_file = 'data/outputs/system_test_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n Detailed report saved to: {report_file}")
    
    return success_rate == 100

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("   SMART SURVEILLANCE SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"\n  Testing all system components...")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_passed = generate_report()
    
    if all_passed:
        print("\n" + "=" * 70)
        print("   SYSTEM READY FOR DEPLOYMENT")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("    PLEASE FIX ISSUES BEFORE DEPLOYMENT")
        print("=" * 70)
