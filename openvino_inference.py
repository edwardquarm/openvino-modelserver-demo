import openvino.runtime as ov  # Updated to avoid deprecation warning
import numpy as np
from PIL import Image
import json
import torch
from torchvision.models import DenseNet161_Weights

def preprocess_image(image_path):
    """
    Preprocess the input image using torchvision's DenseNet161 weights.
    """
    # Load the default weights for DenseNet161
    weights = DenseNet161_Weights.DEFAULT
    preprocess = weights.transforms()
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return weights, image

if __name__ == "__main__":
    core = ov.Core()

    # Compile the Model
    compiled_model = core.compile_model("openvino_model/model.xml", "AUTO")

    # Preprocess the input image
    model_weights, processed_img = preprocess_image("kitten.jpg")

    # Perform inference
    logits = torch.tensor(compiled_model(processed_img)[0]).squeeze(0)

    prediction = logits.softmax(0)  # Apply softmax)

    # Identify the class with the highest score
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = model_weights.meta["categories"][class_id]
    print(f"\nPredicted class: {category_name}, Score: {score:.4f}")
