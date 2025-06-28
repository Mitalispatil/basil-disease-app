import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input as mobilenet_preprocess
from tensorflow.keras.models import load_model
import joblib
import plotly.express as px
from sklearn.decomposition import PCA

st.set_page_config(page_title="🌿 Basil Leaf Disease Classifier", layout="wide")

IMG_SIZE = 224
CATEGORIES = ["Downy mildew", "Fungal", "Fusarium wilt", "Healthy", "Leaf spot"]

# 🎨 CSS
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    h1 { 
        color: #228B22; 
        font-size: 36px; 
        text-align: center;
    }
    .centered-text {
        text-align: center;
    }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; }
    .stPlotlyChart { border: 1px solid #ddd; border-radius: 10px; padding: 5px; background: white; }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Basil Leaf Disease Detection")
st.markdown("<p class='centered-text'>Upload a basil leaf image to classify it using multiple AI models—compare their predictions and confidence levels</p>", unsafe_allow_html=True)

st.markdown("**Supported Disease Classes:**")
st.write(", ".join(CATEGORIES))

# ✅ Load pretrained feature extractors
@st.cache_resource
def load_efficientnet_extractor():
    return EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))

@st.cache_resource
def load_mobilenet_extractor():
    return MobileNet(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Feature dimensionality checker and adapter
def adapt_features(features, expected_dim):
    """Adapts feature dimensions to match what the model expects"""
    features_flat = features.reshape(1, -1)
    current_dim = features_flat.shape[1]
    
    # If dimensions already match, return as is
    if current_dim == expected_dim:
        return features_flat
    
    # If we need to reduce dimensions
    if current_dim > expected_dim:
        # Simple approach: just take the first expected_dim features
        return features_flat[:, :expected_dim]
    
    # If we need to add dimensions (unlikely but for completeness)
    if current_dim < expected_dim:
        # Pad with zeros
        padded = np.zeros((1, expected_dim))
        padded[:, :current_dim] = features_flat
        return padded

# ✅ Load models
@st.cache_resource
def load_main_model():
    return load_model(r"D:\basil_disease_app\efficientnet_models\fine_tuned_efficientnet_model.h5", compile=False)

# @st.cache_resource
# def load_finetuned_efficient_model(): 
#     return joblib.load(r"D:\basil_disease_app\EfficientNet_Models\svm_fine_tuned_efficient.pkl")

# @st.cache_resource
# def load_knn_efficient():
#     return joblib.load(r"D:\basil_disease_app\EfficientNet_Models\knn_fine_tuned_efficient.pkl")

@st.cache_resource
def load_svm_model():
    return joblib.load(r"D:\basil_disease_app\EfficientNet_Models\svm_model.pkl")

@st.cache_resource
def load_mobilenet_model():
    return load_model(r"D:\basil_disease_app\MobileNet_Models\fine_tuned_model.h5", compile=False)

# @st.cache_resource
# def load_knn_mobilenet():
#     return joblib.load(r"D:\basil_disease_app\MobileNet_Models\knn_fine_tuned.pkl")

# Load models and extractors
efficientnet_extractor = load_efficientnet_extractor()
mobilenet_extractor = load_mobilenet_extractor()
main_model = load_main_model()
# finetuned_efficient_model = load_finetuned_efficient_model()
# knn_model_efficient = load_knn_efficient()
svm_model = load_svm_model()
mobilenet_model = load_mobilenet_model()
# knn_model_mobilenet = load_knn_mobilenet()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def safely_predict_with_model(model, features, model_name):
    try:
        # For SVM models
        if hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(features)[0]
            probabilities = 1 / (1 + np.exp(-decision_scores))
            return probabilities
        # For KNN models
        elif hasattr(model, 'predict_proba'):
            return model.predict_proba(features)[0]
        # For neural networks
        else:
            return model.predict(features)[0]
    except Exception as e:
        st.error(f"Error with {model_name}: {str(e)}")
        # Return dummy probabilities
        return np.ones(len(CATEGORIES)) / len(CATEGORIES)  # Equal probabilities

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    image_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image_resized)
    img_batch = np.expand_dims(img_array, axis=0)

    img_preprocessed_eff = efficientnet_preprocess(img_batch)
    img_preprocessed_mob = mobilenet_preprocess(img_batch)

    # Get model expected dimensions
    @st.cache_resource
    def get_model_dimensions():
        # This is a simplified approach - in a real app, you'd 
        # want to determine this programmatically from each model
        dimensions = {
            "finetuned_efficient_model": 128,
            "knn_model_efficient": 128,
            "svm_model": 1280,  # This model expects full dimensions
            "knn_model_mobilenet": 128
        }
        return dimensions
    
    model_dimensions = get_model_dimensions()
    
    # ✅ FEATURE EXTRACTION
    with st.spinner("Extracting features..."):
        # Extract features from EfficientNet
        eff_features = efficientnet_extractor.predict(img_preprocessed_eff)  # shape (1, 1280)
        eff_features_flat = eff_features.reshape(1, -1)  # Keep original full features
        
        # Extract features from MobileNet
        mob_features = mobilenet_extractor.predict(img_preprocessed_mob)  # shape (1, 1024)
        mob_features_flat = mob_features.reshape(1, -1)  # Keep original full features

    # --- Predictions ---
    st.markdown("<h2 class='centered-text'>✅ Predictions from Different Models</h2>", unsafe_allow_html=True)

    # Create columns for better layout with more spacing
    st.markdown("<h3 class='centered-text'>EfficientNetB0 Models</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # 🌟 Main hybrid model - Using full features as it's likely a CNN model
    with col1:
        preds_main = main_model.predict(img_preprocessed_eff)[0]
        idx_main = np.argmax(preds_main)
        label_main = CATEGORIES[idx_main]
        st.success(f"**Finetuned EfficientNetB0 + SVM Hybrid (main):** {label_main} ({preds_main[idx_main]*100:.2f}%)")
        fig_main = px.bar(x=CATEGORIES, y=preds_main*100, labels={'x':'Class','y':'Confidence (%)'}, 
                          title="Finetuned EfficientNetB0 + SVM Hybrid Confidence")
        st.plotly_chart(fig_main, use_container_width=True)

    # # 🌟 Fine-tuned EfficientNet - Using adapted features
    # with col2:
    #     eff_features_adapted = adapt_features(eff_features, expected_dim=model_dimensions["finetuned_efficient_model"])
    #     probabilities_finetuned_eff = safely_predict_with_model(finetuned_efficient_model, 
    #                                                           eff_features_adapted, 
    #                                                           "Fine-tuned EfficientNet")
    #     idx_finetuned_eff = np.argmax(probabilities_finetuned_eff)
    #     label_finetuned_eff = CATEGORIES[idx_finetuned_eff]
    #     st.success(f"**Fine-tuned EfficientNetB0:** {label_finetuned_eff} ({probabilities_finetuned_eff[idx_finetuned_eff]*100:.2f}%)")
    #     fig_finetuned_eff = px.bar(x=CATEGORIES, y=probabilities_finetuned_eff*100, 
    #                               labels={'x':'Class','y':'Confidence (%)'}, 
    #                               title="Fine-tuned EfficientNetB0 Confidence")
    #     st.plotly_chart(fig_finetuned_eff, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    # # 🌟 EfficientNet + KNN - Using adapted features
    # with col3:
    #     #eff_features_adapted_knn = adapt_features(eff_features, expected_dim=model_dimensions["knn_model_efficient"])
    #     preds_knn_eff = safely_predict_with_model(knn_model_efficient, 
    #                                             eff_features_adapted_knn, 
    #                                             "EfficientNet KNN")
    #     idx_knn_eff = np.argmax(preds_knn_eff)
    #     label_knn_eff = CATEGORIES[idx_knn_eff]
    #     st.success(f"**EfficientNet + KNN:** {label_knn_eff} ({preds_knn_eff[idx_knn_eff]*100:.2f}%)")
    #     fig_knn_eff = px.bar(x=CATEGORIES, y=preds_knn_eff*100, 
    #                         labels={'x':'Class','y':'Confidence (%)'}, 
    #                         title="EfficientNet + KNN Confidence")
    #     st.plotly_chart(fig_knn_eff, use_container_width=True)

    # 🌟 EfficientNet + SVM (non-finetuned) - Using FULL features
    with col2:
        # This model needs full 1280 features
        probabilities_svm = safely_predict_with_model(svm_model, 
                                                    eff_features_flat, 
                                                    "EfficientNet SVM (non-finetuned)")
        idx_svm = np.argmax(probabilities_svm)
        label_svm = CATEGORIES[idx_svm]
        st.success(f"**EfficientNetB0 + SVM (non-finetuned):** {label_svm} ({probabilities_svm[idx_svm]*100:.2f}%)")
        fig_svm = px.bar(x=CATEGORIES, y=probabilities_svm*100, 
                        labels={'x':'Class','y':'Confidence (%)'}, 
                        title="EfficientNetB0 + SVM (non-finetuned) Confidence")
        st.plotly_chart(fig_svm, use_container_width=True)

    st.markdown("<h3 class='centered-text'>MobileNetV2 Models</h3>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    # 🌟 MobileNet fine-tuned - Using preprocessed image directly
    with col3:
        preds_mobilenet = mobilenet_model.predict(img_preprocessed_mob)[0]
        idx_mobilenet = np.argmax(preds_mobilenet)
        label_mobilenet = CATEGORIES[idx_mobilenet]
        st.success(f"**Fine-tuned MobileNetV2:** {label_mobilenet} ({preds_mobilenet[idx_mobilenet]*100:.2f}%)")
        fig_mobilenet = px.bar(x=CATEGORIES, y=preds_mobilenet*100, 
                              labels={'x':'Class','y':'Confidence (%)'}, 
                              title="MobileNetV2 Confidence")
        st.plotly_chart(fig_mobilenet, use_container_width=True)

    # # 🌟 MobileNet + KNN - Using adapted features
    # with col6:
    #     mob_features_adapted = adapt_features(mob_features, expected_dim=model_dimensions["knn_model_mobilenet"])
    #     preds_knn_mob = safely_predict_with_model(knn_model_mobilenet, 
    #                                             mob_features_adapted, 
    #                                             "MobileNet KNN")
    #     idx_knn_mob = np.argmax(preds_knn_mob)
    #     label_knn_mob = CATEGORIES[idx_knn_mob]
    #     st.success(f"**MobileNetV2 + KNN:** {label_knn_mob} ({preds_knn_mob[idx_knn_mob]*100:.2f}%)")
    #     fig_knn_mob = px.bar(x=CATEGORIES, y=preds_knn_mob*100, 
    #                         labels={'x':'Class','y':'Confidence (%)'}, 
    #                         title="MobileNetV2 + KNN Confidence")
    #     st.plotly_chart(fig_knn_mob, use_container_width=True)

    # Ensemble prediction (average of all models)
    st.markdown("---")
    st.markdown("<h2 class='centered-text'>🔮 Ensemble Prediction</h2>", unsafe_allow_html=True)
    
    all_predictions = [
        preds_main,
        # probabilities_finetuned_eff,
        # preds_knn_eff,
        probabilities_svm,
        preds_mobilenet,
        # preds_knn_mob
    ]
    
    ensemble_preds = np.mean(all_predictions, axis=0)
    idx_ensemble = np.argmax(ensemble_preds)
    label_ensemble = CATEGORIES[idx_ensemble]
    
    st.success(f"**Ensemble Prediction:** {label_ensemble} ({ensemble_preds[idx_ensemble]*100:.2f}%)")
    
    # Create a larger centered chart for the ensemble
    fig_ensemble = px.bar(x=CATEGORIES, y=ensemble_preds*100, 
                        labels={'x':'Class','y':'Confidence (%)'}, 
                        title="Ensemble Model Confidence")
    fig_ensemble.update_layout(height=400)
    st.plotly_chart(fig_ensemble, use_container_width=True)