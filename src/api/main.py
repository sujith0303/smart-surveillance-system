# REST API for Smart Surveillance System
# File: src/api/main.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import sys
import os
import json
import sqlite3

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.join(project_root, 'src', 'langchain'))

from vector_store import SurveillanceVectorStore
from query_engine import SurveillanceQueryEngine

# Initialize FastAPI
app = FastAPI(
    title="Smart Surveillance API",
    description="Natural language API for surveillance video analysis",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (loaded once at startup)
vector_store = None
query_engine = None
db_path = os.path.join(project_root, 'data/outputs/surveillance.db')
chroma_path = os.path.join(project_root, 'data/outputs/chroma_db')

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global vector_store, query_engine
    
    print("🚀 Initializing Smart Surveillance API...")
    print("📦 Loading vector store...")
    vector_store = SurveillanceVectorStore(chroma_path)
    
    print("🤖 Loading query engine...")
    query_engine = SurveillanceQueryEngine(vector_store)
    
    print("✅ API ready!")

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10

class QueryResponse(BaseModel):
    query: str
    response: str
    result_count: int

class Detection(BaseModel):
    timestamp: float
    object_type: str
    color: str
    confidence: float
    track_id: Optional[int]

class Track(BaseModel):
    track_id: int
    object_type: str
    color: str
    duration: float
    first_seen: float
    last_seen: float
    behavior_count: int

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Smart Surveillance API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/api/query",
            "detections": "/api/detections",
            "tracks": "/api/tracks",
            "track_detail": "/api/tracks/{track_id}",
            "statistics": "/api/statistics",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_store": vector_store is not None,
        "query_engine": query_engine is not None
    }

@app.post("/api/query", response_model=QueryResponse)
async def natural_language_query(request: QueryRequest):
    """
    Natural language query endpoint
    
    Example: POST /api/query
    {
        "query": "Show me people wearing blue",
        "max_results": 10
    }
    """
    if not query_engine:
        raise HTTPException(status_code=503, detail="Query engine not initialized")
    
    try:
        # Execute query
        query_results = query_engine.execute_query(
            request.query, 
            max_results=request.max_results
        )
        
        # Generate response
        response_text = query_engine.generate_response(
            request.query,
            query_results
        )
        
        return QueryResponse(
            query=request.query,
            response=response_text,
            result_count=query_results['result_count']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/detections")
async def get_detections(
    color: Optional[str] = Query(None, description="Filter by color"),
    object_type: Optional[str] = Query(None, description="Filter by object type"),
    min_confidence: Optional[float] = Query(0.5, description="Minimum confidence"),
    limit: Optional[int] = Query(100, description="Max results")
):
    """
    Get filtered detections
    
    Example: GET /api/detections?color=blue&object_type=person&limit=50
    """
    try:
        # Read detections from JSON file
        detections_file = os.path.join(
            project_root, 
            'data/outputs/surveillance_detections.json'
        )
        
        with open(detections_file, 'r') as f:
            all_detections = json.load(f)
        
        # Apply filters
        filtered = all_detections
        
        if color:
            filtered = [d for d in filtered 
                       if d.get('attributes', {}).get('color') == color]
        
        if object_type:
            filtered = [d for d in filtered 
                       if d.get('object_type') == object_type]
        
        if min_confidence:
            filtered = [d for d in filtered 
                       if d.get('confidence', 0) >= min_confidence]
        
        # Limit results
        filtered = filtered[:limit]
        
        # Format response
        results = []
        for d in filtered:
            results.append({
                'timestamp': d.get('timestamp'),
                'object_type': d.get('object_type'),
                'color': d.get('attributes', {}).get('color', 'unknown'),
                'confidence': d.get('confidence'),
                'track_id': d.get('track_id')
            })
        
        return {
            'total': len(results),
            'filters': {
                'color': color,
                'object_type': object_type,
                'min_confidence': min_confidence
            },
            'detections': results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tracks")
async def get_tracks(
    has_behaviors: Optional[bool] = Query(None),
    min_duration: Optional[float] = Query(None),
    object_type: Optional[str] = Query(None),
    limit: Optional[int] = Query(100)
):
    """
    Get tracked objects
    
    Example: GET /api/tracks?has_behaviors=true&min_duration=5.0
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Build query
        query = "SELECT track_id, object_type, color, first_seen, last_seen, duration, total_detections FROM tracks WHERE 1=1"
        params = []
        
        if min_duration is not None:
            query += " AND duration >= ?"
            params.append(min_duration)
        
        if object_type:
            query += " AND object_type = ?"
            params.append(object_type)
        
        query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        tracks = cursor.fetchall()
        
        # Get behavior counts if filtering by behaviors
        results = []
        for track in tracks:
            track_id = track[0]
            
            # Get behavior count
            cursor.execute(
                "SELECT COUNT(*) FROM behaviors WHERE track_id = ?",
                (track_id,)
            )
            behavior_count = cursor.fetchone()[0]
            
            # Apply behavior filter
            if has_behaviors is not None:
                if has_behaviors and behavior_count == 0:
                    continue
                if not has_behaviors and behavior_count > 0:
                    continue
            
            results.append({
                'track_id': track[0],
                'object_type': track[1],
                'color': track[2],
                'first_seen': track[3],
                'last_seen': track[4],
                'duration': track[5],
                'total_detections': track[6],
                'behavior_count': behavior_count
            })
        
        conn.close()
        
        return {
            'total': len(results),
            'filters': {
                'has_behaviors': has_behaviors,
                'min_duration': min_duration,
                'object_type': object_type
            },
            'tracks': results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tracks/{track_id}")
async def get_track_detail(track_id: int):
    """
    Get detailed information for specific track
    
    Example: GET /api/tracks/42
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get track info
        cursor.execute(
            "SELECT * FROM tracks WHERE track_id = ?",
            (track_id,)
        )
        track = cursor.fetchone()
        
        if not track:
            raise HTTPException(status_code=404, detail="Track not found")
        
        # Get behaviors
        cursor.execute(
            "SELECT behavior_type, timestamp, duration, severity FROM behaviors WHERE track_id = ?",
            (track_id,)
        )
        behaviors = cursor.fetchall()
        
        conn.close()
        
        return {
            'track_id': track[0],
            'object_type': track[1],
            'color': track[2],
            'first_seen': track[3],
            'last_seen': track[4],
            'duration': track[5],
            'total_detections': track[6],
            'behaviors': [
                {
                    'type': b[0],
                    'timestamp': b[1],
                    'duration': b[2],
                    'severity': b[3]
                }
                for b in behaviors
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics")
async def get_statistics():
    """
    Get system statistics
    
    Example: GET /api/statistics
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get track count
        cursor.execute("SELECT COUNT(*) FROM tracks")
        total_tracks = cursor.fetchone()[0]
        
        # Get behavior count
        cursor.execute("SELECT COUNT(*) FROM behaviors")
        total_behaviors = cursor.fetchone()[0]
        
        # Get tracks with behaviors
        cursor.execute(
            "SELECT COUNT(DISTINCT track_id) FROM behaviors"
        )
        tracks_with_behaviors = cursor.fetchone()[0]
        
        # Get object type distribution
        cursor.execute(
            "SELECT object_type, COUNT(*) FROM tracks GROUP BY object_type"
        )
        object_distribution = dict(cursor.fetchall())
        
        # Get behavior type distribution
        cursor.execute(
            "SELECT behavior_type, COUNT(*) FROM behaviors GROUP BY behavior_type"
        )
        behavior_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        # Get vector store stats
        vector_stats = vector_store.get_statistics()
        
        return {
            'tracking': {
                'total_tracks': total_tracks,
                'tracks_with_behaviors': tracks_with_behaviors,
                'total_behaviors': total_behaviors,
                'object_distribution': object_distribution,
                'behavior_distribution': behavior_distribution
            },
            'indexing': {
                'total_indexed': vector_stats['total_indexed']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Smart Surveillance API Server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API documentation: http://localhost:8000/docs")
    print("🔍 Alternative docs: http://localhost:8000/redoc")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
