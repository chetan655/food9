import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os



# 👇 Rebuild VGG16 with 9 output classes
def build_model(num_classes=9):
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

base_path = os.path.dirname(__file__)

model_path = os.path.join(base_path, 'model_0.pth')

# model = torch.load(open(model_path, 'rb'))


@st.cache_resource
def load_model():
    checkpoint = torch.load(model_path, map_location="cpu")

    model = build_model(num_classes=9)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

model = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model().to(device)

class_names = ['apple fruit',
 'banana fruit',
 'cherry fruit',
 'chickoo fruit',
 'grapes fruit',
 'kiwi fruit',
 'mango fruit',
 'orange fruit',
 'strawberry fruit']

def transform_image(image):
    test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    return test_transform(image).unsqueeze(0).to(device)





st.title("🍎 Fruit Classifier with Camera")
st.write("Click a picture using your webcam and let the VGG16 model predict the fruit!")

image_data = st.camera_input("Take a picture")

if image_data is not None:
    image = Image.open(image_data).convert("RGB")
    # st.image(image, caption="Captured Image", use_column_width=True)
    st.image(image, caption="Captured Image", use_container_width=True)


    input_tensor = transform_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        # pred = output.argmax().item()
        pred = torch.argmax(output, dim=1).item()

    st.success(f"Prediction: **{class_names[pred]}**")



