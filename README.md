# Supply Chain Demand Forecasting with 1D CNN-LSTM

A state-of-the-art machine learning solution for supply chain demand forecasting using 1D CNN-LSTM hybrid models, part of the research paper titled "Advanced Deep Learning Techniques for Supply Chain Demand Forecasting."

## Overview

This project implements an advanced deep learning approach to predict product demand across multiple retailers in a supply chain network. The main features include:

- Hybrid 1D CNN-LSTM architecture with attention mechanisms
- Multi-scale feature extraction for temporal pattern recognition
- Extensive feature engineering for time series data
- Ensemble learning capabilities
- Comprehensive evaluation metrics with confidence intervals
- Supply chain simulation for generating synthetic data

## Project Structure

supply-chain-forecasting/ ├── analytics/ │ ├── models/ │ │ ├── demand_forecaster.py # Main CNN-LSTM implementation │ │ ├── disruption_predictor.py # Predicts supply chain disruptions │ │ ├── lead_time_predictor.py # Predicts supplier lead times │ │ └── inventory_optimizer.py # Optimizes inventory policies │ └── utils/ │ └── evaluation_report.py # Evaluation metrics and visualizations ├── simulation/ │ ├── supply_chain_simulator.py # Simulates supply chain dynamics │ └── config.py # Configuration parameters ├── output/ │ ├── analytics/ # Analytics results │ │ └── evaluation/ # Evaluation results and visualizations │ └── simulation/ # Simulation data ├── run_pipeline.py # Main execution script ├── requirements.txt # Project dependencies └── README.md



## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:

```bash
git clone https://github.com/parasmahla1/mini-BTP.git
cd supply-chain-forecasting 
```
2. Create and activate a virtual environment (recommended):
```bash 
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n supply-chain-ml python=3.8
conda activate supply-chain-ml
```
3. Install dependencies:
```bash 
pip install -r requirements.txt
```

### Command Line Options

```bash
# Run with CNN-LSTM model (default)
python run_pipeline.py --model cnn_lstm

# Run with XGBoost model instead
python run_pipeline.py --model xgboost

# Specify simulation parameters
python run_pipeline.py --days 365 --suppliers 5 --manufacturers 3 --retailers 10

# Specify forecast horizon
python run_pipeline.py --forecast 30

# Skip simulation and use existing data
python run_pipeline.py --skip-sim

# Only run evaluation on existing results
python run_pipeline.py --eval-only

# Use external indicators data
python run_pipeline.py --external-data path/to/external_data.csv

# Disable ensemble models
python run_pipeline.py --no-ensemble

# Specify CNN-LSTM sequence length
python run_pipeline.py --sequence-length 21
```

#### Advanced Configuration

```bash
# Run with custom CNN-LSTM configuration
python run_pipeline.py --model cnn_lstm --sequence-length 21 --batch-size 32 --epochs 150
```