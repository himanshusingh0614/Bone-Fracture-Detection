# Bone Fracture Classifier

This project is a **Bone Fracture Classification** model implemented in a Jupyter Notebook (`Bone_Fracture_Classifier.ipynb`). The notebook contains a complete pipeline for building and evaluating a machine learning or deep learning model to classify bone fractures from medical images.

## Project Overview

Bone fractures are common medical conditions, and automated classification can assist radiologists in diagnosis. This project leverages image processing and classification techniques to identify fractured bones from medical imaging datasets.

---

## 📂 File Structure

- **Bone_Fracture_Classifier.ipynb**: Main notebook containing the data preprocessing, model training, evaluation, and visualization steps.

---

## 🛠️ Requirements

To run the notebook, the following libraries are required:

- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV (if preprocessing involves image manipulation)
- scikit-learn

You can install the required libraries using:

```bash
pip install tensorflow numpy pandas matplotlib seaborn opencv-python scikit-learn
```

---

## 📊 Dataset

This notebook is designed to work with a dataset of bone x-ray images, where each image is labeled as fractured or non-fractured. Ensure your dataset is organized like this:

```
/dataset/
    /fractured/
        img1.jpg
        img2.jpg
        ...
    /non-fractured/
        img1.jpg
        img2.jpg
        ...
```

---

## ⚙️ Features

- Image Preprocessing: Resizing, normalization, and augmentation.
- Model Creation: Convolutional Neural Network (CNN).
- Training & Validation: Model performance evaluation using accuracy, loss curves, and confusion matrix.
- Visualization: Sample images, training curves, and performance metrics.

---

## 🚀 How to Run

1. Open the notebook in Jupyter or Google Colab.
2. Update the dataset path to point to your dataset location.
3. Run all cells to train and evaluate the model.

---

## 📈 Results

The notebook will output:

- Training and validation accuracy/loss plots.
- Confusion matrix for performance evaluation.
- Example predictions on sample images.

---

## ✨ Future Scope

- Experiment with different CNN architectures.
- Add Grad-CAM visualization to highlight fracture regions.
- Explore transfer learning using pretrained models like VGG16, ResNet, etc.

---

## 📧 Contact

For any queries, feel free to contact the project author.
