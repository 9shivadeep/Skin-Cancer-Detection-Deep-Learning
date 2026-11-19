import json, io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, render_template

app = Flask(__name__, template_folder="templates")
device = torch.device("cpu")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

checkpoint = torch.load("ensemble_trained.pth", map_location=device)
num_classes = checkpoint["num_classes"]

resnet = models.resnet18()
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
resnet.load_state_dict(checkpoint["resnet"])
resnet.eval()

mobilenet = models.mobilenet_v2()
mobilenet.classifier.add_module("1", nn.Linear(1280, num_classes))
mobilenet.load_state_dict(checkpoint["mobilenet"])
mobilenet.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    top_probs = []
    if request.method == "POST":
        file = request.files["file"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            avg_logits = (resnet(tensor) + mobilenet(tensor)) / 2.0
            probs = torch.softmax(avg_logits, dim=1)
        top3_prob, top3_idx = torch.topk(probs, 3)
        flat_idx = top3_idx.view(-1).tolist()
        flat_prob = top3_prob.view(-1).tolist()
        prediction = class_names[flat_idx[0]]
        top_probs = [(class_names[i], round(p * 100, 2)) for i, p in zip(flat_idx, flat_prob)]
    return render_template("index.html", prediction=prediction, top_probs=top_probs)

if __name__ == "__main__":
    app.run(port=5000)
