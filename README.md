## Live Demo
https://your-app-name.onrender.com

# Handwritten Digit Recognizer

End-to-end production ML system for handwritten
digit recognition using CNN trained on MNIST dataset.

## Results
- Test Accuracy  : 99.42%
- Val Accuracy   : 98.78%
- Training Time  : ~5 minutes on CPU
- Model Size     : ~1MB

## Features
- CNN with 99.42% accuracy
- Real time canvas drawing prediction
- FastAPI REST API with auto docs
- Premium glassmorphism UI
- Custom cursor and 3D tilt effects
- Color coded confidence bars
- Auto predict as you draw
- Toggle auto predict on/off
- CLI prediction tool
- Experiment tracking with CSV logs
- Confusion matrix and evaluation charts
- ANN vs CNN comparison

## Tech Stack
- Deep Learning : PyTorch CNN
- API           : FastAPI + Uvicorn
- Frontend      : HTML, CSS, JavaScript
- Config        : PyYAML
- Evaluation    : Scikit-learn, Matplotlib, Seaborn

## Project Structure
digit-recognizer/
├── src/
│   ├── data.py       - Data pipeline
│   ├── model.py      - CNN architecture
│   ├── train.py      - Training loop
│   └── evaluate.py   - Evaluation & charts
├── app/
│   ├── api.py        - FastAPI backend
│   └── predict.py    - Prediction module
├── config/
│   └── config.yaml   - Hyperparameters
├── models/
│   └── mnist_model.pth - Trained model
├── logs/
│   ├── training_log.csv
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── ann_vs_cnn.png
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/script.js
└── requirements.txt

## Setup
git clone https://github.com/VirajK1207/digit-recognizer.git
cd digit-recognizer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Train Model
python -m src.train

## Evaluate Model
python -m src.evaluate

## Run App
python app\api.py

## Open Browser
http://127.0.0.1:8000

## API Docs
http://127.0.0.1:8000/docs

## CLI Tool
python app\predict.py image.png

## Author
Built by Viraj 