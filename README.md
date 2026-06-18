# 🚗 Advanced Smart Parking Detection System

A production-ready, AI-powered parking management system featuring real-time object detection, multi-object tracking, predictive analytics, and intelligent violation detection.

## ✨ Unique Features

- **Advanced Multi-Object Tracking**: DeepSORT + Kalman Filter for robust tracking
- **Real-time Detection**: YOLOv5/v8 with 60+ FPS on GPU
- **Predictive Analytics**: Forecast occupancy 15-60 minutes ahead
- **Violation Detection**: Unauthorized parking, overstay alerts
- **License Plate Recognition**: Optional ALPR integration
- **Heat Maps**: Visualize parking patterns
- **WebSocket Real-time Updates**: Live dashboard streaming
- **RESTful API**: Comprehensive REST endpoints
- **Mobile-Responsive UI**: Modern React dashboard
- **Docker Ready**: Full containerization support
- **CI/CD Pipeline**: GitHub Actions automation
- **Production Monitoring**: Prometheus metrics & logging

## 📋 Quick Start

### Using Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/manvitha30-NM/smart--parking-real-time-object-detection-.git
cd smart--parking-real-time-object-detection-

# Build and run
docker-compose up -d

# Access dashboard at http://localhost:3000
# API at http://localhost:8000
```

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download_models.py

# Start backend
python scripts/start_api.py
```

## 📦 Project Structure

```
smart-parking-detection/
├── src/
│   ├── detection/           # YOLO detection engine
│   ├── tracking/            # DeepSORT tracking
│   ├── analytics/           # Forecasting & analytics
│   ├── parking/             # Space mapping
│   ├── api/                 # FastAPI backend
│   └── utils/               # Utilities
├── frontend/                # React dashboard
├── tests/                   # Unit & integration tests
├── docker/                  # Docker configuration
├── .github/workflows/       # CI/CD pipelines
├── config/                  # Configuration files
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
└── requirements.txt         # Python dependencies
```

## 🎯 Key Components

### 1. Detection Engine
- YOLOv5/v8 based vehicle detection
- 45% confidence threshold
- NMS threshold: 0.45
- Supports GPU acceleration

### 2. Tracking Engine
- DeepSORT algorithm
- Kalman Filter for motion prediction
- Max age: 30 frames
- Min hits: 3 frames

### 3. Analytics Module
- Time-series occupancy forecasting
- Heat map generation
- Peak hours analysis
- Utilization reports

### 4. Violation Detection
- Overstay detection
- Unauthorized parking alerts
- Reserved space violations
- Real-time notifications

### 5. API Server
- FastAPI backend
- RESTful endpoints
- WebSocket for real-time updates
- Prometheus metrics

### 6. Frontend Dashboard
- React-based UI
- Live video feed
- Real-time statistics
- Alert management
- Report generation

## 🚀 API Endpoints

### Base URL
```
http://localhost:8000/api/v1
```

### Key Endpoints

```bash
# Parking Status
GET /parking/status

# Detected Vehicles
GET /vehicles
GET /vehicles/{vehicle_id}

# Analytics
GET /analytics/forecast?horizon=30
GET /analytics/heatmap?date=2024-01-15
GET /analytics/statistics

# Violations
GET /violations
POST /violations/{violation_id}/resolve

# Real-time Stream
WS /stream
```

## 📊 Performance Metrics

| Component | GPU (RTX 3080) | CPU (i7-11700K) |
|-----------|---|---|
| Detection | 60 FPS | 8 FPS |
| Tracking | 50 FPS | 6 FPS |
| Combined | 45 FPS | 5 FPS |
| Memory | 3-4GB | 2GB |
| Latency | <50ms | <200ms |

## 🔧 Configuration

Edit `config/config.yaml`:

```yaml
detection:
  model: "yolov5l"
  confidence: 0.45
  device: "cuda"

tracking:
  max_age: 30
  min_hits: 3

analytics:
  enable_forecasting: true
  forecast_horizon: 30
  enable_heatmaps: true

api:
  host: "0.0.0.0"
  port: 8000
```

## 📝 Requirements

- Python 3.8+
- CUDA 11.0+ (optional, for GPU)
- 8GB RAM minimum
- 500MB disk space

## 📖 Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
- [API Reference](docs/api_reference.md)
- [Configuration Guide](docs/configuration.md)
- [Architecture Details](docs/architecture.md)
- [Deployment Guide](docs/deployment.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 🐳 Docker Deployment

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

## 🔄 CI/CD Pipeline

- **Automated Testing**: PyTest on every push
- **Code Quality**: Black, Flake8, MyPy
- **Security Scanning**: Bandit, Trivy
- **Docker Build**: Automated image building
- **Deployment**: Auto-deploy on main branch

## 📱 Frontend Features

- Real-time video streaming
- Live occupancy dashboard
- Occupancy forecast chart
- Violation alerts
- Heat map visualization
- Report generation & export
- System settings configuration

## 🛠️ Troubleshooting

### GPU Not Detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Model Loading Issues
```bash
python scripts/download_models.py --force
```

### Low Performance
- Reduce input resolution
- Use smaller YOLO model (yolov5s)
- Enable GPU acceleration
- Reduce frame processing rate

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [Deep SORT](https://github.com/nwojke/deep_sort)
- [OpenCV](https://opencv.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/manvitha30-NM/smart--parking-real-time-object-detection-/issues)
- **Discussions**: [GitHub Discussions](https://github.com/manvitha30-NM/smart--parking-real-time-object-detection-/discussions)

---

**⭐ If you find this project helpful, please star it!**
