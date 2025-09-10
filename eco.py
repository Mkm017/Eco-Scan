import streamlit as st
import requests
import time
import os
from PIL import Image
import plotly.express as px
import pandas as pd
import numpy as np

# --- Configuration ---
API_URL = "https://garbage-api-eq5g.onrender.com/classify/"
ST_PAGE_TITLE = "Eco-Scan: AI-Powered Garbage Sorter"
ST_PAGE_ICON = "♻️"
CLASSES_TO_DISPLAY = 5  # Number of top classes to show in the bar chart

DATA_DIR = 'Data'

# Get the list of class names from the folder structure.
# This ensures the predictions are mapped to the correct labels.
if os.path.exists(DATA_DIR):
    class_names = sorted(os.listdir(DATA_DIR))
else:
    class_names = ["Battery", "Biological", "Brown-glass", "Cardboard","Clothes", "Green-glass", "Metal", "Paper", "Plastic", "Shoes", "Trash", "White-glass"]

st.set_page_config(
    page_title=ST_PAGE_TITLE,
    page_icon=ST_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
    }
    .subheader {
        font-size: 1.5rem;
        color: #3CB371;
        border-bottom: 2px solid #3CB371;
        padding-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2E8B57;
        transform: scale(1.05);
    }
    .prediction-card {
        background-color: #f0fff0;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #4CAF50;
    }
    .uploaded-image {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    .uploaded-image:hover {
        transform: scale(1.02);
    }
    .progress-bar {
        height: 1.5rem;
        border-radius: 10px;
        background-color: #e0e0e0;
        margin: 0.5rem 0;
    }
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        transition: width 0.5s ease-in-out;
    }
    .category-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        background-color: #E8F5E9;
        color: #2E8B57;
        font-weight: bold;
        margin: 0.25rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .tutorial-step {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# --- Functions ---
def send_image_to_api(uploaded_file):
    """Sends the uploaded image to the backend API and returns the response."""
    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    try:
        response = requests.post(API_URL, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Connection Error: The backend API is not running. Please start it with `uvicorn key:app --reload`.")
        return None
    except Exception as e:
        st.error(f"An error occurred during API communication: {e}")
        return None

def format_confidence(confidence):
    """Formats confidence to a more realistic percentage value."""
    # Convert to percentage
    percent = confidence * 100
    
    # For very high confidence (>99%), show 2 decimal places but cap at 99.99%
    if percent > 99:
        return min(percent, 99.99)
    
    # For lower confidence, show 1 decimal place
    return round(percent, 1)

# --- Initialize Session State ---
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'result' not in st.session_state:
    st.session_state.result = None
if 'classification_history' not in st.session_state:
    st.session_state.classification_history = []

# --- UI Layout ---
# Header with animated title
st.markdown(f'<h1 class="main-header">{ST_PAGE_ICON} {ST_PAGE_TITLE}</h1>', unsafe_allow_html=True)

# Animated introduction
with st.expander("Welcome to Eco-Classify!", expanded=True):
    st.markdown("""
    **Upload a photo of a waste item**, and our AI will classify it for proper recycling!     
    """)
    
    # Display category tags
    st.markdown("**Supported categories:**")
    categories = class_names
    category_tags = " ".join([f'<span class="category-tag">{cat.capitalize()}</span>' for cat in categories])
    st.markdown(category_tags, unsafe_allow_html=True)

st.markdown("---")

# Main columns for content
col1, col2 = st.columns([1, 2], gap="large")

# Column 1: Image Upload and Display
with col1:
    st.markdown('<p class="subheader">1. Upload an Image</p>', unsafe_allow_html=True)
    
    # Drag and drop area with custom styling
    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG, WEBP",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        # Store in session state
        st.session_state.uploaded_file = uploaded_file
        
        # Display image with styling
        img = Image.open(uploaded_file)
        # Calculate new height to maintain aspect ratio
        w, h = img.size
        new_h = int((300 / w) * h)
        resized_img = img.resize((300, new_h), Image.LANCZOS)
        
        st.markdown('<div class="uploaded-image">', unsafe_allow_html=True)
        st.image(resized_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a clear button
        if st.button("Clear Result", key="clear_btn"):
            st.session_state.uploaded_file = None
            st.session_state.result = None
            st.rerun()

# Column 2: Classification Results and Interactive Elements
with col2:
    st.markdown('<p class="subheader">2. Prediction Results</p>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_file is not None:
        # Classification button with animation
        if st.button("Classify Image", type="secondary", use_container_width=True):
            with st.spinner("Analyzing your image..."):
                # Add a progress bar for visual feedback
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.02)  # Simulate processing time
                    progress_bar.progress(percent_complete + 1)
                
                result = send_image_to_api(st.session_state.uploaded_file)
                
            if result:
                st.session_state.result = result
                # Add to classification history
                st.session_state.classification_history.append({
                    'image': st.session_state.uploaded_file,
                    'result': result,
                    'timestamp': time.time()
                })
                st.success("✅ Classification Complete!")
                
                
    
    if st.session_state.result is not None:
        result = st.session_state.result
        
        predicted_class = result['prediction']
        confidence_score = format_confidence(result['confidence'])
        
        
        # Animated progress bar for confidence
        st.markdown(f"**Confidence:** {confidence_score:.2f}%")
        
        
        
        st.info(f"**Result:** {predicted_class.upper(), 'Please dispose responsibly'}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive visualization of all probabilities
        with st.expander("📊 Detailed Analysis", expanded=True):
            probabilities = result['probabilities']
            sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            
            # Create DataFrame for visualization with formatted percentages
            df = pd.DataFrame(sorted_probs, columns=['Class', 'Probability'])
            df['Probability'] = df['Probability'].apply(format_confidence)
            
            # Create interactive bar chart with Plotly
            fig = px.bar(
                df, 
                x='Class', 
                y='Probability', 
                color='Class',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title="Classification Confidence by Category"
            )
            fig.update_layout(
                xaxis_title="Waste Category",
                yaxis_title="Confidence (%)",
                yaxis_range=[0, 100],  # Set fixed y-axis range for consistency
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data in a table with formatted percentages
            display_df = df.copy()
            display_df = display_df.rename(columns={'Class': 'Category', 'Probability': 'Confidence (%)'})
            display_df['Confidence (%)'] = display_df['Confidence (%)'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

# --- Sidebar with Enhanced Features ---
with st.sidebar:
    st.markdown(f'<h2 style="color: #2E8B57;">{ST_PAGE_ICON} About Eco-Classify</h2>', unsafe_allow_html=True)
    
    # Tutorial steps with expandable sections
    with st.expander("📋 How to Use", expanded=True):
        st.markdown("""
        <div>
        <h4>Step 1: Upload</h4>
        <p>Drag and drop or select an image of a waste item</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div >
        <h4>Step 2: Classify</h4>
        <p>Click the 'Classify Image' button to analyze</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div>
        <h4>Step 3: Review</h4>
        <p>Check the results</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display classification history if available
    if st.session_state.classification_history:
        st.markdown("---")
        st.subheader("📝 Classification History")
        
        # Show last 3 classifications
        for i, history in enumerate(reversed(st.session_state.classification_history[-3:])):
            # Get the image from the history
            img = Image.open(history['image'])
            # Resize for thumbnail display
            w, h = img.size
            new_h = int((150 / w) * h)
            resized_img = img.resize((150, new_h), Image.LANCZOS)
            
            # Format confidence for display
            formatted_confidence = format_confidence(history['result']['confidence'])
            
            with st.expander(f"Classification {len(st.session_state.classification_history)-i}: {history['result']['prediction']}", expanded=False):
                st.image(resized_img, use_container_width=True)
                st.metric(
                    label="Confidence", 
                    value=f"{formatted_confidence:.2f}%"
                )
    
    st.markdown("---")
    # Technical information in an expander
    with st.expander("⚙️ Technical Details"):
        st.info("""
        This app demonstrates a client-server architecture:
        
        - **Frontend**: Streamlit UI (this app)
        - **Backend**: FastAPI server with ML model
        
        The model classifies waste into categories using computer vision.
        
        **Note**: Confidence scores are capped at 99.99% to reflect realistic model performance.
        """)
        
        st.markdown("""
        ### How to Run:
        1. **Start the Backend API:**
           ```bash
           uvicorn key:app --reload
           ```
        2. **Start the Frontend UI:**
           ```bash
           streamlit run eco.py
           ```
        """)
    
    # Add feedback mechanism
    st.markdown("---")
    st.subheader("💬 Feedback")
    feedback = st.text_area("How can we improve Eco-Classify?")
    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback!")
        else:
            st.warning("Please provide feedback before submitting")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Eco-Scan ♻️"
    "</div>",
    unsafe_allow_html=True

)





