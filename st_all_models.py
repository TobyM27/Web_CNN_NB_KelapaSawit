import streamlit as st

import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

import cv2, numpy as np, joblib
from skimage.feature import graycomatrix, graycoprops

# ______________ naive bayes functions ______________

def extract_rgb(image):
    image = image.astype(np.float64)
    
    if image.shape[2] == 4:
        mask = image[:,:,3] > 0
    else:
        mask = np.ones(image.shape[:2], dtype=bool)

    r_mean = np.mean(image[:,:,0][mask])
    g_mean = np.mean(image[:,:,1][mask])
    b_mean = np.mean(image[:,:,2][mask])

    # Normalize
    total = r_mean + g_mean + b_mean
    return (r_mean/total, g_mean/total, b_mean/total)

def extract_texture(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    image = (image / 32).astype(np.uint8)

    glcm = graycomatrix(image, distances=[1], angles=[0, 45, 90, 135], levels=8, symmetric=True, normed=True)

    contrast = np.mean(graycoprops(glcm, 'contrast'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))

    return contrast, correlation, energy, homogeneity

def process_single_image(image):
    r, g, b = extract_rgb(image)

    if image.shape[2] == 4: 
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contrast, correlation, energy, homogeneity = extract_texture(gray_img)

    return np.array([[r, g, b, contrast, correlation, energy, homogeneity]])




# ______________ AlexNet Functions ______________

def load_model():
    num_classes = 5

    model = models.alexnet(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(4096),
        nn.Dropout(p=0.6),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024),
        nn.Dropout(p=0.6),
        nn.Linear(1024, num_classes),
    )

    checkpoint = torch.load('model_lr0.001_optAdam_drop0.6_acc0.8529.pth.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])  

    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0103, 0.0107, 0.0091], 
                           std=[0.0048, 0.0047, 0.0057])
    ])
    return transform(image).unsqueeze(0)

def predict(model, image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()
    



# ______________ ResNet functions ______________

class DaunKelapaSawit_classifier(nn.Module):
    def __init__(self, num_classes=4):
        super(DaunKelapaSawit_classifier, self).__init__()
        self.model = models.resnet34(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    

def preprop(image):
    image = image.resize((224, 224), Image.Resampling.LANCZOS)

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0067, 0.0072, 0.0056], std=[0.0038, 0.0039, 0.0035])
    ])

    return transform(image)
    


# ______________ Main functions ______________
def main():
    st.sidebar.title("Pilihan Model")
    model_choice = st.sidebar.radio(
        "Pilih Model :",
        ('Naive Bayes', 'AlexNet', 'ResNet-34')
    )

    st.title("🌱 Klasifikasi Penyakit Daun Bibit Kelapa Sawit")
    st.write("📷 Upload gambar daun untuk klasifikasi penyakit")


    uploaded_file = st.file_uploader("📂 Pilih gambar", type=['jpg', 'jpeg', 'png'])

    captured_image = st.camera_input("📸 Atau gunakan kamera")

    image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image')
    elif captured_image is not None:
        image = Image.open(captured_image).convert('RGB')
        st.image(image, caption='Captured Image')

    if image:
        if 'Naive Bayes' in model_choice :
            if st.button('Classify'):
                with st.spinner('Processing . . .'):

                    st.image(uploaded_file, caption='Uploaded Image')

                    try:
                        pso_model = joblib.load('naive_bayes_PSO_model.pkl')
                        ga_model = joblib.load('naive_bayes_GA_model.pkl')
                    except:
                        st.error("Error loading models. Please check if model files exist.")
                        return
                    
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

                    features = process_single_image(image)

                    pso_prediction = pso_model[0].predict(features)[0]
                    ga_prediction = ga_model[0].predict(features)[0]

                    st.subheader('Classification Results:')
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("PSO Model Prediction:")
                        st.write(f"**{pso_prediction}**")
                    
                    with col2:
                        st.write("GA Model Prediction:")
                        st.write(f"**{ga_prediction}**")

                    st.subheader("Confidence Scores:")

                    pso_proba = pso_model[0].predict_proba(features)[0]
                    ga_proba = ga_model[0].predict_proba(features)[0]

                    classes = ['bercak_daun', 'daun_berkerut', 'daun_berputar', 'daun_menggulung', 'daun_menguning']

                    col3, col4 = st.columns(2)

                    with col3:
                        st.write("PSO Model Probabilities:")
                        for cls, prob in zip(classes, pso_proba):
                            st.write(f"{cls}: {prob:.4f}")
                            
                    with col4:
                        st.write("GA Model Probabilities:")
                        for cls, prob in zip(classes, ga_proba):
                            st.write(f"{cls}: {prob:.4f}")

        elif 'AlexNet' in model_choice :
            if st.button('Classify'):
                with st.spinner('Processing . . .'):
                    model = load_model()

                    processed_image = preprocess_image(image)
                    prediction = predict(model, processed_image)

                    class_names = ['bercak_daun', 'daun_berkerut', 'daun_berputar', 'daun_menggulung', 'daun_menguning']
                    result = class_names[prediction]

                    st.markdown(f"""<div style="padding: 20px; background-color: #4CAF50; color: white; font-size: 24px; text-align: center; border-radius: 10px;">
                    🚀 **Alexnet Model Prediction: {result}** 🎴 </div>""", unsafe_allow_html=True)

        elif 'ResNet-34' in model_choice :
            if st.button('Classify'):
                with st.spinner('Processing . . .'):
                    saved_model_state = torch.load('model_resnet34_BibitSawit_ADAM_RF_884_10epc_lr01_16bs_20250321_1120.pth', map_location=torch.device('cpu'))

                    saved_model = models.resnet34(weights=None)
                    num_classes = 5
                    saved_model.fc = torch.nn.Linear(in_features=512, out_features=num_classes)

                    saved_model.load_state_dict(saved_model_state)
                    saved_model.eval()

                    class_names = ['bercak_daun', 'daun_berkerut', 'daun_berputar', 'daun_menggulung', 'daun_menguning']

                    mod_image = preprop(image)
                    tensor_img = mod_image.unsqueeze(0)

                    with torch.no_grad():
                        outputs = saved_model(tensor_img)
                        _, preds = torch.max(outputs, 1)
                        class_name = class_names[preds.item()]

                    predicted_idx = preds.item()
                    predicted_class = class_names[predicted_idx]
                    
                    # Calculate probabilities
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

                    st.markdown(f"""<div style="padding: 20px; background-color: #4CAF50; color: white; font-size: 24px; text-align: center; border-radius: 10px;">🚀 **Resnet Model Prediction: {class_name}** 🎴</div>""", unsafe_allow_html=True)

                    # Display class probabilities
                    st.subheader("ResNet-34 Prediction Probabilities:")
                    for i, prob in enumerate(probabilities):
                        probability_percentage = prob.item() * 100
                        st.markdown(f"{class_names[i]}: {probability_percentage:.2f}%")


if __name__ == '__main__' :
    main()