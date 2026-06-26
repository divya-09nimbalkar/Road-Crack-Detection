
---

```markdown
#  Road Crack Detection

A computer vision project for detecting cracks in road surfaces using PyTorch and ResNet18.  
This repository demonstrates an end‑to‑end ML pipeline: data preparation, model training, evaluation, testing, and interactive exploration via Jupyter notebooks.



##  Project Structure

Road_Crack_Detection/

│

├── .venv/                 # Virtual environment

├── data/


│   ├── raw/               # Raw images (dummy or real dataset)

│   └── processed/         # Preprocessed data

├── models/

│   └── resnet18_crack.pth # Saved model weights


│

├── notebooks/

│   └── exploration.ipynb  # Interactive notebook for inference & visualization

│

├── src/

│   ├── __init__.py

│   ├── data_pipeline.py   # Dataset + DataLoader

│   ├── evaluate.py        # Model loading + evaluation functions

│   ├── main.py            # Entry point for training & evaluation

│   ├── model.py           # Model architecture (ResNet18 customization)

│   ├── train.py           # Training loop with accuracy tracking

│   └── utils.py           # Helper utilities

│

├── tests/

│   ├── test_data_pipeline.py

│   ├── test_model.py

│   └── test_utils.py

│

├── .gitignore

├── README.md              # Project documentation

└── requirements.txt       # Python dependencies

```

---

##  Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/divya-09nimbalkar/Road_Crack_Detection.git
   cd Road_Crack_Detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

##  Usage

### Train & Evaluate
Run the pipeline:
```bash
python -m src.main
```

This will:
- Prepare data
- Build the ResNet18 model
- Train for 2 epochs
- Save weights to `models/resnet18_crack.pth`
- Evaluate on sample images

---

### Notebook Exploration
Open the Jupyter notebook:
```bash
jupyter notebook notebooks/exploration.ipynb
```

Features:
- Auto‑train if model file is missing
- Load saved model
- Predict crack/no‑crack on images
- Visualize predictions inline
- Plot training metrics

---

##  Example Output
Training log:
```
Epoch 1/2
Loss: 0.4487 | Accuracy: 0.70
Epoch 2/2
Loss: 0.1917 | Accuracy: 0.80
 Model saved to models/resnet18_crack.pth
```

Notebook prediction:
```
data/raw/road_0.png → Crack
```

---

##  Next Steps
- Integrate real datasets (e.g., SDNET2018, CrackForest)
- Add data augmentation (resize, normalize, random flips)
- Track precision, recall, F1‑score
- Deploy as a Streamlit app for interactive use
- Expand unit tests in `tests/` for reproducibility

---

##  Tech Stack
- Python 3.11
- PyTorch & TorchVision
- OpenCV
- tqdm
- Jupyter Notebook
- pytest (for tests)

---

##  Author
Developed by **Divya**  
AI/ML Developer | Full‑stack Python | Computer Vision Enthusiast


