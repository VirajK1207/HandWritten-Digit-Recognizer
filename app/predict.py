# app/predict.py - Prediction Module

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import base64
import io
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from src.model import get_model

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def load_model():
    model      = get_model()
    checkpoint = torch.load(
        config["paths"]["best_model"],
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model loaded successfully!")
    print(f"Saved at epoch   : {checkpoint['epoch']}")
    print(f"Val Accuracy     : {checkpoint['val_acc']:.2f}%")
    return model


def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")

    # Convert to numpy array
    img_array = np.array(image)

    # If background is white invert it
    if img_array.mean() > 127:
        img_array = 255 - img_array

    # Normalize to full 0-255 range
    if img_array.max() > 0:
        img_array = (img_array / img_array.max() * 255).astype(np.uint8)

    # Threshold - make digit pure white
    img_array = np.where(img_array > 30, 255, 0).astype(np.uint8)

    # Convert back to PIL
    image = Image.fromarray(img_array)

    # Apply transforms
    tensor = transform(image).unsqueeze(0)
    return tensor


def decode_base64_image(base64_str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes))
    return image


def predict(model, image):
    tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = F.softmax(outputs, dim=1)[0]

    predicted_digit = probs.argmax().item()
    confidence      = probs[predicted_digit].item() * 100

    all_probs = {
        str(i): round(probs[i].item() * 100, 2)
        for i in range(10)
    }

    return {
        "digit":      predicted_digit,
        "confidence": round(confidence, 2),
        "all_probs":  all_probs
    }


def predict_from_base64(model, base64_str):
    image = decode_base64_image(base64_str)
    return predict(model, image)


def predict_from_file(model, file_path):
    image = Image.open(file_path)
    return predict(model, image)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python app/predict.py image.png")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    print(f"Loading model...")
    model = load_model()

    print(f"Predicting: {image_path}")
    result = predict_from_file(model, image_path)

    print(f"\nPredicted Digit : {result['digit']}")
    print(f"Confidence      : {result['confidence']:.2f}%")
    print(f"\nAll Probabilities:")
    for digit, prob in sorted(
        result["all_probs"].items(),
        key=lambda x: -x[1]
    ):
        bar = "█" * int(prob / 5)
        print(f"  Digit {digit}: {bar} {prob:.2f}%")