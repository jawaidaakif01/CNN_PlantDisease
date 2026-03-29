🌿 Crop Disease AI Diagnostician
📌 Project Overview
Agriculture faces a massive threat from crop diseases, often resulting in significant yield losses before symptoms are even identified by the human eye. This project implements a Deep Learning solution to provide instant, automated diagnosis of 15 different plant health states (including bacterial spots, blights, and viruses) across Tomato, Potato, and Pepper crops.

🛠️ Tech Stack & Engineering Highlights
Deep Learning Framework: TensorFlow / Keras

Architecture: MobileNetV2 (Transfer Learning)

Optimization: Data Augmentation (Rotation, Zoom, Flips) & Dropout (0.5) to combat overfitting.

Performance: Achieved 91.16% Validation Accuracy on a 15-class classification problem.

🧠 The Engineering Journey
1. The Challenge: Overfitting
Initial custom CNN models reached 98% training accuracy but stalled at 87% validation accuracy. This indicated the model was "memorizing" specific leaf images rather than learning general disease features.

2. The Solution: Transfer Learning & Augmentation
To solve this, I pivoted to a Transfer Learning approach using MobileNetV2, pre-trained on the ImageNet dataset. This provided the model with a "PhD-level" understanding of textures and edges.

Data Augmentation: I implemented a preprocessing pipeline that randomly flipped and rotated images during training, ensuring the model never saw the exact same pixel layout twice.

Dropout: Added a 50% Dropout layer to the classification head to force redundant feature learning.

3. Results
The final model successfully generalized to unseen data, maintaining a stable 91.16% accuracy and a significantly lower loss compared to the baseline custom architecture.

📂 Project Structure
app.py: The Streamlit web interface code.

Model_Training.ipynb: The complete Google Colab training history, including data cleaning and loss curves.

mobilenet_crop_disease.keras: The final exported weights and architecture.

requirements.txt: Necessary libraries for deployment.

💻 Local Installation
Clone the repo:
git clone https://github.com/jawaidaakif01/CNN_PlantDisease.git

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py

📈 Future Scope
Expand the dataset to include more crop varieties like Corn and Wheat.

Deploy as a lightweight mobile app using TensorFlow Lite.

Integrate GPS tagging for localized disease heatmaps for farmers.
