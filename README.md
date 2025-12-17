<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Streamlit-1.44+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">CardioScan AI</h1>
<h3 align="center">ECG Analysis & Heart Disease Detection</h3>

<p align="center">
  An AI-powered web application that analyzes ECG (Electrocardiogram) images to detect cardiovascular diseases using deep learning.
</p>

---

## Overview

**CardioScan AI** is a medical imaging application built with Streamlit and PyTorch that leverages a fine-tuned ResNet-18 neural network to classify ECG images into four categories:

| Category | Description |
|----------|-------------|
| **Normal** | Healthy heart rhythm with no abnormalities detected |
| **Abnormal Heartbeat** | Irregular rhythm patterns indicating arrhythmia |
| **History of MI** | Signs of a previous myocardial infarction (heart attack) |
| **Myocardial Infarction** | Active signs of a heart attack |

---

## Features

- **Deep Learning Analysis** - Powered by a fine-tuned ResNet-18 model trained on ECG data
- **12-Lead ECG Extraction** - Automatically segments and analyzes all 12 standard ECG leads
- **1D Signal Visualization** - Converts ECG images to waveform signals for detailed analysis
- **Instant Predictions** - Get results in under 5 seconds
- **Camera Capture** - Upload images or capture directly using your device camera
- **Modern UI** - Clean, professional medical-grade interface

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core programming language |
| **PyTorch** | Deep learning framework |
| **Streamlit** | Web application framework |
| **OpenCV** | Image processing |
| **NumPy / Pandas** | Data manipulation |
| **Matplotlib** | Signal visualization |
| **Pillow** | Image handling |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/harishkumbarSs/Cardiovascular_Disease_Detection.git
   cd Cardiovascular_Disease_Detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   - **Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   streamlit run app/streamlit_app.py
   ```

6. **Open in browser**
   
   Navigate to `http://localhost:8501`

---

## Project Structure

```
Cardiovascular_Disease_Detection/
├── app/
│   └── streamlit_app.py      # Main Streamlit application
├── models/
│   └── best_model.pth        # Trained PyTorch model
├── notebooks/                 # Jupyter notebooks for experimentation
├── src/
│   ├── __init__.py
│   ├── data_transforms.py    # Image transformation pipelines
│   ├── split_dataset.py      # Dataset splitting utilities
│   ├── train.py              # Model training script
│   └── utils/
│       └── image_processing.py  # ECG image processing functions
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## Usage

1. **Launch the application** and navigate to the **ECG Analysis** tab
2. **Upload an ECG image** (PNG, JPG, JPEG) or capture one using your camera
3. **View processing steps:**
   - Grayscale conversion
   - 12-lead extraction
   - 1D signal waveforms
4. **Get AI prediction** with confidence scores for each condition

---

## Model Information

- **Architecture:** ResNet-18 (transfer learning from ImageNet)
- **Classes:** 4 (Normal, Abnormal Heartbeat, History of MI, Myocardial Infarction)
- **Input:** 224x224 RGB ECG images
- **Framework:** PyTorch

---

## Disclaimer

> **This application is for educational and research purposes only.**
> 
> It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for accurate diagnosis and treatment recommendations.

---

## Author

**Harish Kumbar**

- GitHub: [@harishkumbarSs](https://github.com/harishkumbarSs)

---


