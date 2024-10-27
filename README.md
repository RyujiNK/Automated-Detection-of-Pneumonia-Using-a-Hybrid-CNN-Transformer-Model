# Automated Detection of Pneumonia Using a Hybrid CNN-Transformer Model

This repository hosts the code and resources for the final project in the Data-Driven Life Sciences course, which aims to develop a hybrid CNN-Transformer model to automatically classify chest X-ray images for pneumonia detection. By combining Convolutional Neural Networks (CNNs) and Transformer layers, the model captures both local and global features to enhance diagnostic accuracy and interpretability.

## Table of Contents
- [Project Overview](#project-overview)
- [Background and Motivation](#background-and-motivation)
- [Dataset](#dataset)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Limitations and Future Directions](#limitations-and-future-directions)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Project Overview
Pneumonia is a leading cause of death worldwide, especially in vulnerable populations. This project uses deep learning techniques to assist radiologists by automatically identifying pneumonia in chest X-ray images. The model leverages the strengths of CNNs for local feature extraction and Transformers for modeling long-range dependencies, creating a robust hybrid model (ConViT) that provides accurate and explainable predictions.

## Background and Motivation
Early detection of pneumonia through chest X-ray imaging can significantly improve patient outcomes. Traditional CNNs are effective for this task but are limited by their focus on local patterns, potentially missing critical global context in medical images. This project implements a hybrid approach using CNN-Transformer architecture to improve classification performance, offering a more comprehensive diagnostic tool for clinical use.

## Dataset
- **Dataset Used**: PneumoniaMNIST from the MedMNIST repository
- **Source**: [MedMNIST Repository](https://github.com/MedMNIST/MedMNIST)
- **Description**: The dataset contains 5,856 labeled chest X-ray images, with balanced classes for pneumonia and healthy cases.
- **Preprocessing**: Images were normalized and resized to 128x128 pixels. Data augmentation (random flipping, rotation, and cropping) was applied to increase robustness.
- **Data Split**: 70% training, 15% validation, 15% testing, with cross-validation for robust model evaluation.

## Installation
To get started, clone this repository:
```bash
git clone https://github.com/yourusername/Pneumonia_Detection_Using_a_Hybrid_CNN-Transformer_Model.git
```
### Prerequisites
- Python 3.7+
- Jupyter Notebook or Google Colab access
- Required packages are listed in `requirements.txt`

Install dependencies:
```bash
pip install -r requirements.txt
```

### Running in Google Colab
Open the `Pneumonia_Detection_Using_a_Hybrid_CNN-Transformer_Model.ipynb` file in Google Colab for an interactive environment with all dependencies preloaded. The code will guide you through loading the dataset, training the model, and evaluating results.

## How to Run
1. **Data Loading and Preprocessing**: Load the PneumoniaMNIST dataset, normalize images, and apply data augmentations.
2. **Model Training**: Follow the notebook to initialize and train the CNN-Transformer (ConViT) model. Model checkpoints and early stopping are used to optimize training.
3. **Evaluation**: The notebook includes scripts for evaluating accuracy, F1 score, precision, recall, and generating attention maps for explainability.

To run the code locally, open the notebook in Jupyter Notebook or JupyterLab:
```bash
jupyter notebook Pneumonia_Detection_Using_a_Hybrid_CNN-Transformer_Model.ipynb
```

## Model Architecture
The hybrid CNN-Transformer (ConViT) model architecture includes:
1. **CNN Backbone**: Extracts low-level features from the X-ray images.
2. **Transformer Layers**: Encodes global context by analyzing relationships across different image regions.
3. **Fully Connected Layer**: Classifies images as pneumonia or healthy.

**Attention Maps**: Transformer layers generate attention maps, highlighting areas of focus for better interpretability.

## Results
- **Performance Metrics**: Model achieves high accuracy, F1 score, and ROC-AUC, outperforming traditional CNN-only models.
- **Explainability**: Attention maps reveal image regions influencing predictions, aiding radiologists in understanding model decisions.

Results are documented in the `results` folder, with saved models and evaluation metrics.

## Limitations and Future Directions
- **Current Limitations**: Requires labeled data and may struggle with data heterogeneity across different populations.
- **Future Directions**: Explore multi-modal data integration (e.g., clinical metadata), additional training on larger datasets, and incorporating uncertainty estimation.

## Contributing
Contributions to this project are welcome! For major changes, please open an issue first to discuss the proposed modifications.

## Acknowledgments
This project was developed by Bestun Altuni and Natthaphong Kaewkam as part of the Data-Driven Life Sciences course. Special thanks to MedMNIST for the dataset and the OpenAI API for additional resources.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

This README provides a comprehensive guide for new users and contributors, ensuring they can set up, understand, and use the project effectively. Let me know if you need any specific modifications!
