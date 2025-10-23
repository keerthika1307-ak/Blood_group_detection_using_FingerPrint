import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Blood Group Detection",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background: white;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    h1 {
        color: #667eea;
        text-align: center;
        font-size: 3em;
        font-weight: bold;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        padding: 10px 30px;
        font-weight: bold;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Function to preprocess the image
def preprocess_image(img, target_size=(64, 64)):
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load Model
@st.cache_resource
def load_model():
    model_path = "model/bloodgroup_cnn_model.keras"
    if not os.path.exists(model_path):
        st.error("⚠️ Model not found. Please train the model first.")
        st.stop()
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading the model: {str(e)}")
    st.stop()

# Class Labels
# Derive from dataset folder to ensure correct label ordering
try:
    if os.path.isdir("dataset"):
        class_names = sorted([
            d for d in os.listdir("dataset")
            if os.path.isdir(os.path.join("dataset", d))
        ])
        if len(class_names) == 0:
            class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
    else:
        class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
except Exception:
    class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Sidebar Information
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4320/4320350.png", width=100)
    st.title("ℹ️ About")
    st.markdown("""
    ### Blood Group Detection System
    
    This AI-powered application uses **Convolutional Neural Networks (CNN)** 
    to predict blood groups from fingerprint images.
    
    #### 📊 Model Performance:
    - Overall Accuracy: **59.6%**
    - Best performing: B-, A+, O+
    
    #### 🎯 Supported Blood Groups:
    - A+, A-, B+, B-
    - AB+, AB-, O+, O-
    
    #### 📁 Accepted Formats:
    - JPG, JPEG, PNG, BMP
    
    #### 💡 Tips:
    - Use clear fingerprint images
    - Ensure good lighting
    - Images should be focused
    """)
    
    st.divider()
    st.markdown("**Made with ❤️ using Streamlit & TensorFlow**")

# Main Content
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1>🩸 Blood Group Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em; color: #666;'>Upload a fingerprint image to discover your blood group</p>", unsafe_allow_html=True)

st.divider()

# Upload Section
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a fingerprint image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a clear fingerprint image for best results"
    )
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="✅ Uploaded Image", use_container_width=True)

with col_right:
    if uploaded_file is not None:
        with st.spinner("🔍 Analyzing fingerprint..."):
            try:
                # Preprocess and predict
                variants = [
                    img,
                    ImageOps.mirror(img),
                    ImageOps.flip(img),
                    img.rotate(10, resample=Image.BILINEAR),
                    img.rotate(-10, resample=Image.BILINEAR),
                    ImageOps.autocontrast(img, cutoff=2),
                    ImageOps.autocontrast(ImageOps.mirror(img), cutoff=2)
                ]
                preds = []
                for v in variants:
                    arr = preprocess_image(v)
                    p = model.predict(arr, verbose=0)
                    preds.append(p)
                prediction = np.mean(preds, axis=0)
                
                # Get predictions
                predicted_idx = np.argmax(prediction[0])
                predicted_class = class_names[predicted_idx]
                confidence = float(prediction[0][predicted_idx]) * 100
                
                # Display main prediction
                st.markdown("### 🎯 Prediction Result")
                st.markdown(f"""
                <div class='prediction-box'>
                    Blood Group: {predicted_class}<br>
                    <span style='font-size: 0.6em;'>Confidence: {confidence:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence indicator
                if confidence > 70:
                    st.success("✅ High Confidence Prediction")
                elif confidence > 50:
                    st.warning("⚠️ Moderate Confidence - Results may vary")
                else:
                    st.error("❌ Low Confidence - Consider using a clearer image")

                # Top-3 predictions
                top3_idx = np.argsort(prediction[0])[::-1][:3]
                st.markdown("#### Top-3 Predictions")
                for rank, idx in enumerate(top3_idx, start=1):
                    st.write(f"{rank}. {class_names[idx]} — {prediction[0][idx]*100:.1f}%")
                
            except Exception as e:
                st.error(f"❌ Error processing the image: {str(e)}")

# Probability Distribution
if uploaded_file is not None:
    st.divider()
    st.markdown("### 📊 Probability Distribution")
    
    # Create probability chart
    probabilities = [prediction[0][i] * 100 for i in range(len(class_names))]
    
    # Create two columns for chart and table
    chart_col, table_col = st.columns([2, 1])
    
    with chart_col:
        # Plotly bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=class_names,
                y=probabilities,
                marker=dict(
                    color=probabilities,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f'{p:.1f}%' for p in probabilities],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Blood Group Prediction Probabilities",
            xaxis_title="Blood Group",
            yaxis_title="Probability (%)",
            height=400,
            showlegend=False,
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with table_col:
        st.markdown("#### Detailed Probabilities")
        for i, class_name in enumerate(class_names):
            prob = prediction[0][i] * 100
            st.metric(label=class_name, value=f"{prob:.2f}%")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.info("🔬 **AI-Powered Analysis**")
with col2:
    st.info("⚡ **Real-Time Results**")
with col3:
    st.info("🎯 **59.6% Accuracy**")