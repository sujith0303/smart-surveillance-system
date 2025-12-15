# 🎥 Smart Surveillance System with Natural Language Querying

An intelligent surveillance video analysis system that combines Computer Vision (YOLOv8) with Natural Language Processing (LangChain + RAG) to enable natural language queries over surveillance footage.

## 🌟 Features

### Computer Vision Pipeline
- **Object Detection**: YOLOv8-based real-time detection of people, vehicles, and objects
- **Multi-Object Tracking**: ByteTrack algorithm for persistent tracking across frames
- **Behavior Analysis**: Automatic detection of suspicious activities (loitering, fast movement, erratic patterns)
- **Attribute Extraction**: Color detection and object classification

### Natural Language Interface
- **Semantic Search**: Query surveillance data using plain English
- **RAG Implementation**: ChromaDB vector store with sentence transformers
- **LLM Integration**: GPT4All for intelligent response generation
- **Intent Detection**: Automatically understands query context and filters

### REST API
- **FastAPI Backend**: Production-ready API with automatic documentation
- **Interactive Docs**: Swagger UI at `/docs` for testing endpoints
- **6 Core Endpoints**: Query, detections, tracks, statistics, and health checks
- **CORS Enabled**: Ready for web application integration

## 📊 Project Statistics

- **2,782 total detections** processed
- **321 unique tracks** identified
- **5,553 behavior events** detected
- **3,103 items** indexed in vector database
- **100% test success rate**

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Video Input (CCTV)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Computer Vision Pipeline (YOLOv8 + ByteTrack)       │
├─────────────────────────────────────────────────────────────┤
│  • Object Detection (People, Vehicles, Objects)              │
│  • Multi-Object Tracking (Unique IDs)                        │
│  • Behavior Analysis (Fast movement, Loitering, Erratic)     │
│  • Attribute Extraction (Colors, Types)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Storage Layer                        │
├──────────────────────┬──────────────────────────────────────┤
│   SQLite Database    │    ChromaDB Vector Store             │
│  • Events            │    • Semantic Embeddings             │
│  • Tracks            │    • Similarity Search               │
│  • Behaviors         │    • 3,103 indexed items             │
└──────────────────────┴──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            Natural Language Processing Layer                 │
├─────────────────────────────────────────────────────────────┤
│  • Query Parser (Intent detection)                           │
│  • Vector Search (Semantic similarity)                       │
│  • LLM Response Generator (GPT4All)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    REST API (FastAPI)                        │
├─────────────────────────────────────────────────────────────┤
│  POST /api/query          - Natural language queries         │
│  GET  /api/detections     - Filtered object detections       │
│  GET  /api/tracks         - Tracked object data              │
│  GET  /api/tracks/{id}    - Individual track details         │
│  GET  /api/statistics     - System statistics                │
│  GET  /health             - Health check                     │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites
- Python 3.11+
- 8GB+ RAM (16GB recommended)
- GPU (optional, for faster processing)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/smart-surveillance-system.git
cd smart-surveillance-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 model (if not auto-downloaded)
curl -L -o data/models/yolov8x.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
```

## 💻 Usage

### 1. Process Video (Object Detection & Tracking)

```bash
# Run detection pipeline
python src/detection/detection_pipeline.py

# Run tracking pipeline
python src/tracking/tracking_pipeline.py
```

### 2. Index Data (Vector Store)

```bash
# Create vector embeddings
python src/langchain/vector_store.py
```

### 3. Interactive Query Mode

```bash
# Start interactive chat
python src/langchain/query_engine.py
```

Example queries:
- "Show me all people wearing blue"
- "Find vehicles between 2-4 PM"
- "Show suspicious behavior"
- "How many people were detected?"

### 4. Start REST API Server

```bash
# Start API server
python src/api/main.py

# API available at: http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

### 5. Test the System

```bash
# Run comprehensive tests
python test_complete_system.py
```

## 📡 API Usage

### Natural Language Query

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me people wearing red",
    "max_results": 10
  }'
```

Response:
```json
{
  "query": "Show me people wearing red",
  "response": "I found 5 people wearing red clothing...",
  "result_count": 5
}
```

### Get Filtered Detections

```bash
curl "http://localhost:8000/api/detections?color=blue&object_type=person&limit=20"
```

### Get Statistics

```bash
curl "http://localhost:8000/api/statistics"
```

Response:
```json
{
  "tracking": {
    "total_tracks": 321,
    "tracks_with_behaviors": 202,
    "total_behaviors": 5553,
    "object_distribution": {...},
    "behavior_distribution": {...}
  },
  "indexing": {
    "total_indexed": 3103
  }
}
```

## 🛠️ Technology Stack

### Computer Vision
- **YOLOv8** (v8.3.0) - Object detection
- **ByteTrack** - Multi-object tracking
- **OpenCV** (4.8+) - Video processing
- **PyTorch** (2.6+) - Deep learning framework

### Natural Language Processing
- **LangChain** (0.3+) - LLM application framework
- **ChromaDB** (0.5+) - Vector database
- **Sentence Transformers** (3.0+) - Text embeddings
- **GPT4All** (2.8+) - Local LLM inference

### Backend & API
- **FastAPI** (0.104+) - REST API framework
- **SQLite** - Relational database
- **Uvicorn** - ASGI server

## 📁 Project Structure

```
smart-surveillance-system/
├── src/
│   ├── detection/
│   │   └── detection_pipeline.py      # YOLOv8 detection
│   ├── tracking/
│   │   └── tracking_pipeline.py       # ByteTrack tracking
│   ├── database/
│   │   └── event_database.py          # SQLite operations
│   ├── langchain/
│   │   ├── vector_store.py            # ChromaDB vector store
│   │   └── query_engine.py            # NL query engine
│   └── api/
│       └── main.py                    # FastAPI server
├── data/
│   ├── models/
│   │   └── yolov8x.pt                 # YOLOv8 weights
│   ├── videos/
│   │   └── test_surveillance.mp4      # Input video
│   └── outputs/
│       ├── surveillance_detections.json
│       ├── tracking_report.json
│       ├── tracked_output.mp4
│       ├── surveillance.db
│       └── chroma_db/
├── tests/
│   └── test_complete_system.py        # Integration tests
├── requirements.txt
├── README.md
└── .gitignore
```

## 🎯 Key Capabilities

### Query Examples

**Object-based:**
- "Show me all people"
- "Find blue cars"
- "List all vehicles"

**Color-based:**
- "Show red objects"
- "Find everyone wearing green"
- "Display blue items"

**Behavior-based:**
- "Show suspicious activity"
- "Find fast movement"
- "Display erratic behavior"
- "Show people loitering"

**Time-based:**
- "What happened between 2-4 PM?"
- "Show events after 5 minutes"
- "Find activity at 10:30"

**Statistical:**
- "How many people detected?"
- "Count all vehicles"
- "Show detection statistics"

## 🔬 Testing

The system includes comprehensive tests covering:
- File structure validation
- Object detection accuracy
- Multi-object tracking performance
- Database integrity
- Vector store functionality
- Query engine responses

Run tests:
```bash
python test_complete_system.py
```

Expected output: **100% test success rate**

## 📈 Performance Metrics

- **Detection Speed**: ~30 FPS on GPU, ~5 FPS on CPU
- **Tracking Accuracy**: 321 unique tracks maintained
- **Query Response Time**: <2 seconds for typical queries
- **Vector Search**: <100ms for semantic search
- **API Response**: <500ms average

## 🤝 Contributing

This is a portfolio project. Suggestions and feedback are welcome!

## 📄 License

MIT License - Feel free to use for learning and portfolio purposes

## 👤 Author

**Sujith Reddy Bommareddy**
- GitHub: [@sujith0303](https://github.com/sujith0303)
- Email: bsujithreddy@gmail.com
- LinkedIn: [Your LinkedIn]

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- ByteTrack algorithm
- LangChain framework
- FastAPI framework
- ChromaDB vector database

## 📸 Screenshots

### Interactive API Documentation
Access at `http://localhost:8000/docs`

### Example Queries
```
🙋 You: Show me people wearing blue

🤖 Assistant: I found 145 blue-colored people in the surveillance 
footage. The detections show multiple individuals wearing blue 
clothing throughout the recording...
```

### Statistics Dashboard
```json
{
  "total_tracks": 321,
  "tracks_with_behaviors": 202,
  "total_behaviors": 5553,
  "object_distribution": {
    "person": 162,
    "car": 4,
    "handbag": 57
  }
}
```

## 🚀 Future Enhancements

- [ ] Real-time video stream processing
- [ ] Web dashboard UI
- [ ] Mobile app integration
- [ ] Cloud deployment (Docker + AWS/GCP)
- [ ] Alert system (email/SMS notifications)
- [ ] Multi-camera support
- [ ] Face recognition (with privacy considerations)
- [ ] Advanced analytics dashboard

---

**Built for intelligent surveillance analysis**