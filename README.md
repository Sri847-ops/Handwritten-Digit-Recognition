# Handwritten Digit Recognition ✍️

Welcome to the **Handwritten Digit Recognition** project! This system is built using **TensorFlow** and **Keras** to classify handwritten digits from the **MNIST dataset**.

## 🛠 Tech Stack

- **Framework**: TensorFlow, Keras
- **Dataset**: MNIST (handwritten digits)
- **Development**: Jupyter Notebook
- **Model**: Fully Connected Neural Network (Dense Layers)

## 🔧 Getting Started

### Clone the Repository
```sh
git clone https://github.com/your-username/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Run the Jupyter Notebook
```sh
jupyter notebook Handwritten_Digit_Recognition.ipynb
```

## 🧠 Model Architecture
The neural network consists of:
- **Flatten layer** to convert 28x28 images into 1D vectors
- **Hidden layer** with 500 neurons and ReLU activation
- **Output layer** with 10 neurons and softmax activation

## 🚀 Usage
After training, you can test the model using:
```python
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('handwritten_digit_model.h5')

# Example: Predict a single digit (modify as needed)
prediction = model.predict(np.expand_dims(test_image.flatten(), axis=0))
print("Predicted Digit:", np.argmax(prediction))
```

## 📊 Results
- The trained model achieves an accuracy of approximately **97-98%** on the test dataset.
- Sample predictions can be viewed in the notebook after evaluation.

## 🤝 Contribution
We welcome contributions! Follow these steps to contribute:

### 📌 How to Contribute
- **Fork** this repository.
- **Create a new branch** (`feature-branch-name`).
- **Commit your changes** (`git commit -m "Add feature XYZ"`).
- **Push to your fork** and submit a **Pull Request**.
