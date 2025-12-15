# Week 2 Day 8-10: Vector Database & Embeddings Setup
# File: src/langchain/vector_store.py

import json
import os
from typing import List, Dict, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class SurveillanceVectorStore:
    """
    Vector database for semantic search of surveillance events
    Converts events to embeddings for natural language queries
    """
    
    def __init__(self, persist_directory='data/outputs/chroma_db'):
        """
        Initialize vector store with ChromaDB
        
        Args:
            persist_directory: Where to save the vector database
        """
        self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="surveillance_events",
            metadata={"description": "Surveillance video events and tracks"}
        )
        
        # Initialize sentence transformer for embeddings
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded!")
        
        self.event_counter = 0
    
    def create_event_description(self, event: Dict) -> str:
        """
        Create natural language description of event for embedding
        
        Args:
            event: Event dictionary
            
        Returns:
            Human-readable description string
        """
        descriptions = []
        
        # Time description
        timestamp = event.get('timestamp', 0)
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        time_desc = f"at {minutes} minutes {seconds} seconds"
        
        # Object description
        obj_type = event.get('object_type', 'object')
        color = event.get('attributes', {}).get('color', '')
        
        if color and color != 'unknown':
            descriptions.append(f"{color} {obj_type}")
        else:
            descriptions.append(obj_type)
        
        # Add action/context
        if obj_type == 'person':
            descriptions.append("person appeared")
        elif obj_type in ['car', 'truck', 'motorcycle', 'bus']:
            descriptions.append("vehicle detected")
        
        # Track ID if available
        track_id = event.get('track_id')
        if track_id:
            descriptions.append(f"track ID {track_id}")
        
        # Combine into sentence
        full_description = f"A {' '.join(descriptions)} {time_desc}"
        
        return full_description
    
    def create_track_description(self, track: Dict) -> str:
        """
        Create natural language description of track
        
        Args:
            track: Track dictionary
            
        Returns:
            Human-readable description
        """
        track_id = track.get('track_id', 'unknown')
        obj_type = track.get('attributes', {}).get('type', 'object')
        color = track.get('attributes', {}).get('color', '')
        duration = track.get('duration', 0)
        first_seen = track.get('first_seen', 0)
        behaviors = track.get('behaviors', [])
        
        # Build description
        parts = []
        
        # Object description
        if color and color != 'unknown':
            parts.append(f"{color} {obj_type}")
        else:
            parts.append(obj_type)
        
        # Track ID
        parts.append(f"with track ID {track_id}")
        
        # Timing
        minutes = int(first_seen // 60)
        seconds = int(first_seen % 60)
        parts.append(f"first seen at {minutes} minutes {seconds} seconds")
        
        # Duration
        parts.append(f"tracked for {duration:.1f} seconds")
        
        # Behaviors
        if behaviors:
            behavior_types = list(set([b['type'] for b in behaviors]))
            if behavior_types:
                parts.append(f"showed behaviors: {', '.join(behavior_types)}")
        
        description = "A " + " ".join(parts)
        
        return description
    
    def index_detection_results(self, detection_json_path: str):
        """
        Index detection results from JSON file
        
        Args:
            detection_json_path: Path to detection JSON file
        """
        print(f"📥 Loading detection results from: {detection_json_path}")
        
        with open(detection_json_path, 'r') as f:
            events = json.load(f)
        
        print(f"📊 Found {len(events)} events to index")
        
        # Prepare data for batch insertion
        documents = []
        metadatas = []
        ids = []
        
        for i, event in enumerate(events):
            # Create description
            description = self.create_event_description(event)
            documents.append(description)
            
            # Create metadata
            metadata = {
                'type': 'detection',
                'timestamp': float(event.get('timestamp', 0)),
                'object_type': event.get('object_type', 'unknown'),
                'color': event.get('attributes', {}).get('color', 'unknown'),
                'confidence': float(event.get('confidence', 0)),
                'frame_number': int(event.get('frame_number', 0)),
                'track_id': int(event.get('track_id', -1)) if event.get('track_id') else -1
            }
            metadatas.append(metadata)
            
            # Unique ID
            ids.append(f"detection_{i}")
        
        # Batch insert
        print("🔄 Creating embeddings and indexing...")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Indexed {len(events)} detection events")
        self.event_counter += len(events)
    
    def index_tracking_results(self, tracking_json_path: str):
        """
        Index tracking results from JSON file
        
        Args:
            tracking_json_path: Path to tracking JSON file
        """
        print(f"📥 Loading tracking results from: {tracking_json_path}")
        
        with open(tracking_json_path, 'r') as f:
            report = json.load(f)
        
        tracks = report.get('tracks', [])
        print(f"📊 Found {len(tracks)} tracks to index")
        
        # Prepare data
        documents = []
        metadatas = []
        ids = []
        
        for track in tracks:
            # Create description
            description = self.create_track_description(track)
            documents.append(description)
            
            # Create metadata
            metadata = {
                'type': 'track',
                'track_id': int(track.get('track_id', -1)),
                'object_type': track.get('attributes', {}).get('type', 'unknown'),
                'color': track.get('attributes', {}).get('color', 'unknown'),
                'duration': float(track.get('duration', 0)),
                'first_seen': float(track.get('first_seen', 0)),
                'last_seen': float(track.get('last_seen', 0)),
                'has_behaviors': len(track.get('behaviors', [])) > 0,
                'behavior_count': len(track.get('behaviors', []))
            }
            metadatas.append(metadata)
            
            # Unique ID
            ids.append(f"track_{track['track_id']}")
        
        # Batch insert
        print("🔄 Creating embeddings and indexing...")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Indexed {len(tracks)} tracks")
        self.event_counter += len(tracks)
    
    def search(self, query: str, n_results: int = 10, 
               filter_dict: Optional[Dict] = None) -> Dict:
        """
        Semantic search through indexed events
        
        Args:
            query: Natural language search query
            n_results: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            Search results dictionary
        """
        print(f"\n🔍 Searching for: '{query}'")
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )
        
        print(f"✅ Found {len(results['documents'][0])} results")
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0],
            'ids': results['ids'][0]
        }
    
    def search_by_time_range(self, query: str, start_time: float, 
                            end_time: float, n_results: int = 10) -> Dict:
        """
        Search within specific time range
        
        Args:
            query: Natural language query
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            n_results: Number of results
            
        Returns:
            Filtered search results
        """
        filter_dict = {
            "$and": [
                {"timestamp": {"$gte": start_time}},
                {"timestamp": {"$lte": end_time}}
            ]
        }
        
        return self.search(query, n_results, filter_dict)
    
    def search_by_object_type(self, query: str, object_type: str, 
                             n_results: int = 10) -> Dict:
        """
        Search for specific object type
        
        Args:
            query: Natural language query
            object_type: Type of object (person, car, etc)
            n_results: Number of results
            
        Returns:
            Filtered search results
        """
        filter_dict = {"object_type": object_type}
        return self.search(query, n_results, filter_dict)
    
    def search_by_color(self, color: str, n_results: int = 10) -> Dict:
        """
        Search for objects of specific color
        
        Args:
            color: Color to search for
            n_results: Number of results
            
        Returns:
            Filtered search results
        """
        query = f"objects with {color} color"
        filter_dict = {"color": color}
        return self.search(query, n_results, filter_dict)
    
    def get_statistics(self) -> Dict:
        """Get statistics about indexed data"""
        total_count = self.collection.count()
        
        # This is a simplified version - ChromaDB doesn't have built-in aggregations
        return {
            'total_indexed': total_count,
            'collection_name': self.collection.name
        }
    
    def clear_database(self):
        """Clear all data from vector store"""
        print("⚠️  Clearing vector database...")
        self.client.delete_collection("surveillance_events")
        self.collection = self.client.get_or_create_collection(
            name="surveillance_events",
            metadata={"description": "Surveillance video events and tracks"}
        )
        self.event_counter = 0
        print("✅ Database cleared")


# Usage Example and Testing
if __name__ == "__main__":
    print("=" * 70)
    print("🎯 VECTOR DATABASE SETUP & TESTING")
    print("=" * 70)
    
    # Initialize vector store
    vector_store = SurveillanceVectorStore('data/outputs/chroma_db')
    
    # Index detection results
    vector_store.index_detection_results('data/outputs/surveillance_detections.json')
    
    # Index tracking results
    vector_store.index_tracking_results('data/outputs/tracking_report.json')
    
    # Get statistics
    stats = vector_store.get_statistics()
    print(f"\n📊 Vector Database Statistics:")
    print(f"   Total indexed items: {stats['total_indexed']}")
    
    # Test searches
    print("\n" + "=" * 70)
    print("🔍 TESTING SEMANTIC SEARCH")
    print("=" * 70)
    
    # Test 1: Search for people
    print("\n1️⃣ Search: 'people walking'")
    results = vector_store.search("people walking", n_results=5)
    for i, (doc, meta, dist) in enumerate(zip(results['documents'], 
                                               results['metadatas'], 
                                               results['distances'])):
        print(f"\n   Result {i+1}:")
        print(f"   📝 {doc}")
        print(f"   🏷️  Type: {meta['object_type']}, Color: {meta['color']}")
        print(f"   📏 Relevance: {1 - dist:.3f}")
    
    # Test 2: Search by color
    print("\n2️⃣ Search: 'blue objects'")
    results = vector_store.search_by_color("blue", n_results=5)
    for i, (doc, meta) in enumerate(zip(results['documents'], 
                                        results['metadatas'])):
        print(f"\n   Result {i+1}:")
        print(f"   📝 {doc}")
        print(f"   🏷️  {meta['object_type']}")
    
    # Test 3: Search for vehicles
    print("\n3️⃣ Search: 'vehicles or cars'")
    results = vector_store.search("vehicles or cars", n_results=5)
    for i, (doc, meta) in enumerate(zip(results['documents'], 
                                        results['metadatas'])):
        print(f"\n   Result {i+1}:")
        print(f"   📝 {doc}")
        print(f"   🏷️  {meta['object_type']}")
    
    # Test 4: Search for suspicious behavior
    print("\n4️⃣ Search: 'suspicious or unusual behavior'")
    results = vector_store.search("suspicious or unusual behavior", n_results=5)
    for i, (doc, meta) in enumerate(zip(results['documents'], 
                                        results['metadatas'])):
        print(f"\n   Result {i+1}:")
        print(f"   📝 {doc}")
        if meta.get('has_behaviors'):
            print(f"   ⚠️  Behaviors detected: {meta['behavior_count']}")
    
    print("\n" + "=" * 70)
    print("✅ VECTOR DATABASE SETUP COMPLETE!")
    print("=" * 70)