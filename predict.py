import gradio as gr
from torchvision import transforms
import torch
from torchvision.models import mobilenet_v2, resnet18
import os
import config
from torch.nn.functional import softmax

def predict(image):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config.model == 'resnet18':
        model = resnet18(num_classes=config.num_classes).to(device)
        checkpoint = torch.load('./runs/20251206162718-resnet18-bs64-lr0.05-ep30/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = mobilenet_v2(num_classes=config.num_classes).to(device)
        checkpoint = torch.load('./runs/20251206180214-mobilenet_v2-bs64-lr0.05-ep30/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    class_names = os.listdir(os.path.join(config.data_dir, 'test'))

    return {class_names[i]: float(softmax(output, dim=1)[0][i]) for i in range(len(class_names))}
    

gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=26)).launch()

