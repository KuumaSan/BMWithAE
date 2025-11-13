# BMWithAE - Bias Mitigation with Accuracy Enhancement

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Flask Version](https://img.shields.io/badge/flask-3.0.0-green)
![License](https://img.shields.io/badge/license-MIT-orange)

An interactive visual analytics system for bias mitigation and accuracy enhancement in machine learning models.

[Features](#features) • [Quick Start](#quick-start) • [Documentation](#documentation) • [API](#api-reference)

</div>

---

## Overview

BMWithAE is a comprehensive visual analytics system designed to help data scientists and researchers identify, analyze, and mitigate bias in machine learning models while maintaining or improving model accuracy. The system provides an intuitive web interface for exploring data distributions, analyzing fairness metrics, and executing iterative debiasing processes.

## Features

### Core Capabilities

**Multidimensional Bias Analysis**
- Statistical Parity
- Equal Opportunity
- Equalized Odds
- Disparate Impact

**Interactive Data Exploration**
- Dynamic feature distribution visualization
- Subgroup analysis with detailed bias metrics
- Multiple protected attribute selection
- Real-time bias scoring

**Iterative Debiasing Process**
- Bias Mitigation (BM) module
- Accuracy Enhancement (AE) module
- Real-time progress monitoring
- Historical iteration tracking

**Rich Visualizations**
- Dual-chart real-time updates
- Max Epsilon convergence tracking
- Accuracy evolution curves
- Modal dialogs for detailed metrics

### Technical Stack

- **Backend**: Python, Flask, Pandas, Scikit-learn
- **Frontend**: Native HTML/CSS/JavaScript (no build required)
- **Machine Learning**: Multiple classifiers (XGBoost, LightGBM, CatBoost)
- **Data Formats**: Excel (.xlsx), CSV
- **Architecture**: RESTful API design

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository

```bash
git clone https://github.com/KuumaSan/BMWithAE.git
cd BMWithAE
```

2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Start the backend server

```bash
cd backend
python app.py
```

The backend will start at `http://localhost:5000`

5. Open the frontend

Open `frontend/index.html` in your browser, or serve it via a local server:

```bash
cd frontend
python -m http.server 8000
# Visit http://localhost:8000
```

## Documentation

### Project Structure

```
BMWithAE/
├── backend/              # Backend Python code
│   ├── app.py           # Flask main application
│   ├── core_config.py   # Core configuration
│   ├── module_BM.py     # Bias mitigation module
│   ├── module_AE.py     # Accuracy enhancement module
│   ├── eval.py          # Evaluation module
│   └── ...
├── frontend/            # Frontend code
│   ├── index.html       # Main interface
│   ├── styles.css       # Stylesheet
│   └── api.js           # API client
├── data/                # Data files
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Usage Guide

**Step 1: Load Data**

Click "Load Demo" to load the sample dataset, or click "Upload Data" to upload your own dataset.

**Step 2: Explore Data**

- Select protected attributes (multiple selection supported)
- View bias scores and detailed metrics
- Click on features to see distributions and subgroup analysis

**Step 3: Configure Parameters**

Click "Configuration" to set debiasing parameters including classifier selection, iteration count, and threshold values.

**Step 4: Execute Debiasing**

- "Run All Steps": Automatically execute all iterations
- "Step by Step": Execute one iteration at a time for detailed observation

**Step 5: View Results**

- Left chart: Max Epsilon / fairness metrics evolution
- Right chart: Accuracy evolution
- Below: Iteration history records

## API Reference

### Data Management

- `POST /api/upload` - Upload new dataset
- `GET /api/datasets` - List available datasets
- `GET /api/data/<dataset_id>/info` - Get dataset details
- `POST /api/data/<dataset_id>/bias-metrics` - Calculate bias metrics

### Debiasing Process

- `POST /api/debias/init` - Initialize debiasing job
- `POST /api/debias/<job_id>/step` - Execute one iteration
- `GET /api/debias/<job_id>/status` - Get job status
- `POST /api/debias/<job_id>/stop` - Stop job

### Configuration

- `GET /api/config` - Get current configuration
- `POST /api/config` - Update configuration

For detailed API documentation, see [PROJECT_SETUP.md](PROJECT_SETUP.md)

## Configuration

Main configuration parameters in `backend/core_config.py`:

```python
PARAMS_MAIN_MAX_ITERATION = 20        # Maximum iterations
PARAMS_MAIN_THRESHOLD_EPSILON = 0.9   # Epsilon threshold
PARAMS_MAIN_CLASSIFIER = 'XGB'        # Classifier selection
PARAMS_MAIN_TRAINING_RATE = 0.7       # Training set ratio
```

## Troubleshooting

### Port Conflict

Modify the port in `backend/backend_config.py`:

```python
PORT = 5001  # Change to another port
```

### Dependency Installation Issues

```bash
# Upgrade pip
pip install --upgrade pip

# Install problematic packages separately
pip install xgboost --no-cache-dir
```

### CORS Errors

Ensure the backend is running and Flask-CORS is properly configured.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions:

- Submit an Issue: [GitHub Issues](https://github.com/KuumaSan/BMWithAE/issues)
- Email: your.email@example.com

## Acknowledgments

- Inspired by FairSight and FairVis visual analytics systems
- Based on fairness in machine learning research
- Thanks to the open-source community

---

<div align="center">

Made with care for Fair ML

</div>
