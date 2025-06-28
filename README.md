# Basil Leaf Disease Classification 🌿🦠

A deep learning-based web application to detect and classify basil leaf diseases using hybrid models. This tool is designed to assist farmers in identifying plant diseases early, ensuring better yield and crop health.

## 🚀 Project Overview

This project classifies basil plant leaves into five disease categories:
- Healthy
- Downy Mildew
- Fusarium Wilt
- Fungal Infection
- Leaf Spot

We used **EfficientNetB0** and **MobileNetV2** as feature extractors, paired with traditional ML classifiers (SVM, KNN, Random Forest). The best performing model was **EfficientNetB0 + SVM**, achieving an accuracy of **~98%**.

The final model is integrated into a **Streamlit web app**, where users can upload leaf images to get instant predictions.

---

## 📊 Dataset

- The dataset consists of basil leaf images across 5 classes mentioned above.
- Each image is labeled based on the type of disease or if it's healthy.
- **Note:** The dataset is not included in this repository due to size constraints. You may upload your own labeled basil leaf dataset with a similar structure:

---

## 🧠 Models Used

- **EfficientNetB0 + SVM** ✅ (Best Accuracy: ~98%)
- MobileNetV2 + SVM
- EfficientNetB0 + KNN
- EfficientNetB0 + Random Forest
- MobileNetV2 + KNN
- MobileNetV2 + Random Forest

---

## 💻 Tech Stack

- **Python**
- **TensorFlow/Keras**
- **Scikit-learn**
- **Jupyter Notebook**
- **Streamlit** 

---

## 🌐 Web App Features

- Upload an image of a basil leaf.
- Get instant disease prediction.
- Simple and user-friendly interface built using Streamlit.

---

## Install dependencies:
  pip install -r requirements.txt

## Run the Streamlit app:
  streamlit run app.py

## 🧑‍💻 Author

**Mitali Patil**  
[GitHub Profile](https://github.com/Mitalispatil)  
