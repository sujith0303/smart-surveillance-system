# Project Summary

## Quick Stats
- **Total Detections**: 2,782
- **Unique Tracks**: 321
- **Behavior Events**: 5,553
- **Indexed Items**: 3,103
- **Test Success Rate**: 100%

## Key Features
1. YOLOv8 Object Detection
2. ByteTrack Multi-Object Tracking
3. Behavior Analysis (fast movement, loitering, erratic patterns)
4. Natural Language Queries (ChromaDB + LangChain + GPT4All)
5. REST API (FastAPI with Swagger docs)

## Quick Commands

### Run Detection
```bash
python src/detection/detection_pipeline.py
```

### Run Tracking
```bash
python src/tracking/tracking_pipeline.py
```

### Start API
```bash
python src/api/main.py
```

### Interactive Query
```bash
python src/langchain/query_engine.py
```

### Run Tests
```bash
python test_complete_system.py
```

## API Endpoints
- `POST /api/query` - Natural language queries
- `GET /api/detections` - Filtered detections
- `GET /api/tracks` - Track information
- `GET /api/statistics` - System statistics
- `GET /health` - Health check
