import torch
import torchvision.models as models
import cv2

def evaluate_model(net):
    print("Evaluating model...")
    # Load one of the dummy images
    img_path = "data/raw/road_0.png"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float() / 255.0

    outputs = net(img_tensor)
    _, preds = torch.max(outputs, 1)
    print(f"Predictions: {preds}")
    print(f"✅ Loaded image {img_path} with shape {img.shape}")

def load_model(path="models/resnet18_crack.pth"):
    # Build the same architecture used in training
    net = models.resnet18(weights=None)  # no pretrained weights
    net.fc = torch.nn.Linear(net.fc.in_features, 2)  # binary classification
    net.load_state_dict(torch.load(path))
    net.eval()
    print(f"✅ Model loaded from {path}")
    return net
