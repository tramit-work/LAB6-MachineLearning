from flask import Flask, request, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("D:/Downloads/dataMLP/MLP_dress.pth", map_location=device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),  
    transforms.Grayscale(),       
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

def predict(image):
    image = transform(image).unsqueeze(0).to(device) 
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_number():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    image = Image.open(file.stream)
    prediction = predict(image)
    
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)