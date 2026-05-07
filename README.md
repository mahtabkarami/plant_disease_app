# Plant Disease Detection App

A Deep Learning-based web application for plant disease classification using plant leaf images.

The application allows users to upload images and receive disease predictions through an interactive Streamlit interface.  
The model is trained on a multi-class dataset containing 38 plant disease categories.

---

## Features

- Plant disease classification using Deep Learning
- Image upload and real-time prediction
- Streamlit-based user interface
- TensorFlow/Keras model inference
- Confidence score prediction
- Support for 38 disease classes

---

## Dataset

The model is trained on a plant disease dataset with 38 classes, including:

- Tomato diseases
- Corn diseases
- Apple diseases
- Potato diseases
- Grape diseases
- Pepper diseases
- Healthy plant categories

Dataset Link: https://www.kaggle.com/datasets/emmarex/plantdisease

---

## Tech Stack

### Machine Learning
- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib

### Web Application
- Streamlit

---

## Project Structure

```bash
plant_disease_app/
│
├── app.py
├── model/
├── labels.txt
├── requirements.txt
├── images/
├── LICENSE
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/mahtabkarami/plant_disease_app.git

cd plant_disease_app
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Live Demo

Add your deployed Streamlit application link here:

```text
https://plantdiseaseapp-5qvk8vpqecuwsjkgcc3mqn.streamlit.app/
```

---

## Workflow

1. Upload a plant leaf image  
2. The image is preprocessed  
3. The model performs prediction  
4. The predicted disease class is displayed  
5. Confidence score is generated  

---

## Project Goal

This project explores the use of Deep Learning and Computer Vision for early plant disease detection and agricultural monitoring.

---

## License

This project is licensed under the MIT License.

---

## Author

Mahtab Karami  

GitHub: https://github.com/mahtabkarami  
LinkedIn: https://www.linkedin.com/in/mahtab-karami-052658249?utm_source=share_via&utm_content=profile&utm_medium=member_android
Live Demo: https://plantdiseaseapp-5qvk8vpqecuwsjkgcc3mqn.streamlit.app/
