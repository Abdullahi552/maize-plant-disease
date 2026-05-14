# ============================================================================
# MAIZE LEAF DISEASE DETECTION - STREAMLIT APP (FIXED VERSION)
# ============================================================================

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Maize Disease Classifier",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1B5E20, #4CAF50);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #E8F5E9;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    .result-card {
        background: linear-gradient(135deg, #1B5E20, #2E7D32);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .result-card h2 {
        margin: 0;
        font-size: 2rem;
    }
    .result-card p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
    }
    .info-box {
        background: #F5F5F5;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .disease-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        cursor: pointer;
        transition: transform 0.2s;
    }
    .disease-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FF9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #F44336;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        color: #666;
        padding: 1rem;
        margin-top: 2rem;
        border-top: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and class names
@st.cache_resource
def load_model():
    """Load the traced MobileViT model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "mobilevit_maize_model_traced.pt"
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None
    
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model, device

@st.cache_data
def load_class_names():
    """Load class names from JSON file"""
    class_path = "class_names.json"
    
    if not os.path.exists(class_path):
        st.error(f"Class names file not found: {class_path}")
        return None
    
    with open(class_path, "r") as f:
        return json.load(f)

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(model, image_tensor, device, class_names):
    """Run prediction on the image"""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class_idx].item()
        
    return {
        'class': class_names[predicted_class_idx],
        'class_index': predicted_class_idx,
        'confidence': confidence,
        'probabilities': {class_names[i]: probabilities[0][i].item() for i in range(len(class_names))}
    }

# Disease information database
disease_info = {
    "Blight": {
        "description": "Northern Corn Leaf Blight is a fungal disease caused by Exserohilum turcicum. It is one of the most destructive maize diseases worldwide, especially in humid growing regions.",
        "symptoms": "Long, elliptical, gray-green or tan lesions that run parallel to the leaf veins. Lesions can grow up to 10 inches long and may merge together, causing the entire leaf to die.",
        "treatment": "Apply fungicides containing azoxystrobin, pyraclostrobin, or propiconazole at early stages. Remove and destroy infected crop debris. Practice crop rotation with non-host crops.",
        "prevention": "Plant resistant hybrids. Maintain proper plant spacing for good air circulation. Avoid overhead irrigation. Apply foliar fungicides when conditions favor disease development.",
        "image": "🌿"
    },
    "Common_Rust": {
        "description": "Common Rust is caused by the fungus Puccinia sorghi. It occurs worldwide and can cause significant yield losses when infection occurs early in the growing season.",
        "symptoms": "Circular to elongated cinnamon-brown pustules on both upper and lower leaf surfaces. Pustules turn dark brown to black as they mature. Severely infected leaves may turn yellow and die.",
        "treatment": "Apply fungicides at the first sign of infection. Use products containing triazole or strobilurin fungicides. In severe cases, multiple applications may be needed.",
        "prevention": "Plant resistant hybrids. Avoid planting in areas with history of rust. Early planting can help avoid peak disease pressure. Monitor fields regularly during the growing season.",
        "image": "🍂"
    },
    "Gray_Leaf_Spot": {
        "description": "Gray Leaf Spot is caused by Cercospora zeae-maydis. It is a serious foliar disease that can cause significant yield losses, particularly in continuous corn production areas.",
        "symptoms": "Small, necrotic spots that develop into long, rectangular lesions (1-6 inches) with parallel sides. Lesions are gray to tan with dark borders. Severely infected leaves may die prematurely.",
        "treatment": "Apply fungicides at tasseling stage. Use products containing strobilurin and triazole fungicides. Till crop residue to speed decomposition.",
        "prevention": "Plant resistant hybrids. Practice crop rotation with soybeans or other non-host crops. Maintain balanced fertility. Reduce plant density to improve air circulation.",
        "image": "🍃"
    },
    "Healthy": {
        "description": "Healthy maize leaf showing no signs of disease infection. The plant is growing under optimal conditions with proper nutrition and management.",
        "symptoms": "No visible symptoms of disease. Leaf is uniformly green with healthy tissue structure. No lesions, spots, or discoloration present.",
        "treatment": "Continue good agricultural practices. Maintain proper irrigation and fertilization. Monitor regularly for early signs of disease.",
        "prevention": "Maintain proper plant nutrition. Practice integrated pest management. Use certified disease-free seeds. Keep fields free from weeds that may harbor diseases.",
        "image": "🌽"
    }
}

# Header
st.markdown("""
    <div class="main-header">
        <h1>🌽 Maize Leaf Disease Detection System</h1>
        <p>Powered by MobileViT Deep Learning Model | 96.18% Accuracy</p>
    </div>
""", unsafe_allow_html=True)

# Load model and class names
with st.spinner("Loading model..."):
    model, device = load_model()
    class_names = load_class_names()

if model is None or class_names is None:
    st.error("""
    ⚠️ **Required files not found!**
    
    Please ensure the following files are in the same directory as this app:
    - `mobilevit_maize_model_traced.pt`
    - `class_names.json`
    
    Also ensure the `test_images` folder contains sample images.
    """)
    st.stop()

st.success(f"✅ Model loaded successfully! | Device: {device} | Classes: {len(class_names)}")

# Sidebar
with st.sidebar:
    st.markdown("### 📋 About")
    st.info("""
    This system uses a **MobileViT (Mobile Vision Transformer)** 
    deep learning model to classify maize leaf diseases from images.
    
    **Model Performance:**
    - 🎯 Accuracy: **96.18%**
    - 📊 Precision: **96.41%**
    - 🔍 Recall: **96.18%**
    - ⭐ F1-Score: **96.24%**
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Disease Classes")
    for cls in class_names:
        icon = disease_info.get(cls, {}).get("image", "🌿")
        st.markdown(f"- {icon} **{cls}**")
    
    st.markdown("---")
    st.markdown("### 📊 How to Use")
    st.markdown("""
    1. **Upload** an image of a maize leaf
    2. Or **select a sample** image from the gallery
    3. Click **'Predict Disease'**
    4. View results and get **treatment recommendations**
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - ✅ Use clear, well-lit images
    - ✅ Center the leaf in the frame
    - ✅ Avoid blurry or dark images
    - ✅ Capture clear disease symptoms
    - ✅ Use high-resolution images when possible
    """)

# Main content - Two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload or Select Image")
    
    # Option to upload or use sample
    option = st.radio("Choose input method:", ["📁 Upload Image", "🖼️ Use Sample Image"], horizontal=True)
    
    uploaded_file = None
    image = None
    image_source = None
    
    if option == "📁 Upload Image":
        uploaded_file = st.file_uploader("Choose a maize leaf image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            # FIXED: Changed use_container_width to use_column_width
            st.image(image, caption="Uploaded Image", use_column_width=True)
            image_source = "uploaded"
    else:
        # Get sample images from test_images folder
        sample_images = []
        sample_paths = []
        
        if os.path.exists("test_images"):
            for f in os.listdir("test_images"):
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    # Store original filename and display name
                    sample_images.append(f.replace('.jpg', '').replace('_', ' '))
                    sample_paths.append(os.path.join("test_images", f))
        
        if sample_images:
            # Create a grid of sample images
            st.markdown("**Select a sample image:**")
            
            # Use columns for better layout
            cols = st.columns(min(2, len(sample_images)))
            selected_idx = None
            
            for idx, (name, path) in enumerate(zip(sample_images, sample_paths)):
                col_idx = idx % 2
                with cols[col_idx]:
                    img = Image.open(path).convert('RGB')
                    # FIXED: Changed use_container_width to use_column_width
                    st.image(img, caption=name, use_column_width=True)
                    if st.button(f"Select {name}", key=f"sample_{idx}"):
                        selected_idx = idx
            
            if selected_idx is not None:
                image = Image.open(sample_paths[selected_idx]).convert('RGB')
                image_source = "sample"
                st.success(f"✅ Selected: {sample_images[selected_idx]}")
                # Show the selected image
                st.image(image, caption=f"Selected: {sample_images[selected_idx]}", use_column_width=True)
        else:
            st.warning("No sample images found. Please upload an image or add images to the 'test_images' folder.")

with col2:
    st.markdown("### 🔍 Prediction Results")
    
    if st.button("🌽 Predict Disease", type="primary", use_container_width=True):
        if image is not None:
            with st.spinner("🔬 Analyzing image..."):
                # Preprocess and predict
                image_tensor = preprocess_image(image)
                result = predict(model, image_tensor, device, class_names)
                
                # Determine confidence class for styling
                confidence_class = "confidence-high" if result['confidence'] > 0.8 else "confidence-medium" if result['confidence'] > 0.6 else "confidence-low"
                
                # Display result card
                icon = disease_info.get(result['class'], {}).get("image", "🌿")
                st.markdown(f"""
                    <div class="result-card">
                        <h2>{icon} {result['class']}</h2>
                        <p>Confidence: <span class="{confidence_class}">{result['confidence']:.2%}</span></p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display probability bar chart
                st.markdown("#### 📊 Probability Distribution")
                prob_df = pd.DataFrame({
                    'Disease': list(result['probabilities'].keys()),
                    'Probability': list(result['probabilities'].values())
                })
                
                fig = px.bar(prob_df, x='Disease', y='Probability', 
                            title="Class Probabilities",
                            labels={'Probability': 'Confidence Score', 'Disease': 'Disease Class'},
                            color='Probability',
                            color_continuous_scale='RdYlGn',
                            range_color=[0, 1])
                fig.update_layout(height=350, showlegend=False)
                fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                
                # Display disease information
                st.markdown("#### 📚 Disease Information")
                info = disease_info.get(result['class'], disease_info["Healthy"])
                
                with st.expander("📖 Description", expanded=True):
                    st.write(info['description'])
                
                with st.expander("⚠️ Symptoms", expanded=True):
                    st.write(info['symptoms'])
                
                with st.expander("💊 Treatment Recommendations", expanded=True):
                    st.write(info['treatment'])
                
                with st.expander("🛡️ Prevention Strategies", expanded=True):
                    st.write(info['prevention'])
        else:
            st.error("⚠️ Please upload an image or select a sample image first.")

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p>🌽 Maize Disease Detection System | Powered by MobileViT | Academic Research Project</p>
        <p style="font-size: 0.8rem;">For best results, use clear images with good lighting. Always consult with agricultural experts for final diagnosis.</p>
    </div>
""", unsafe_allow_html=True)

# Display model performance metrics in sidebar if expanded
with st.sidebar.expander("📊 Model Performance Details"):
    st.markdown("""
    | Metric | Value |
    |--------|-------|
    | Test Accuracy | **96.18%** |
    | Weighted Precision | **96.41%** |
    | Weighted Recall | **96.18%** |
    | Weighted F1-Score | **96.24%** |
    | Macro Precision | **94.45%** |
    | Macro Recall | **95.64%** |
    | Macro F1-Score | **94.95%** |
    
    **Class-wise Performance:**
    | Class | Precision | Recall | F1-Score |
    |-------|-----------|--------|-----------|
    | Blight | 96.41% | 91.48% | 93.88% |
    | Common_Rust | 98.98% | 98.48% | 98.73% |
    | Gray_Leaf_Spot | 82.42% | 92.59% | 87.21% |
    | Healthy | 100.00% | 100.00% | 100.00% |
    """)