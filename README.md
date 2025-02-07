# **Named Entity Recognition (NER) Model for Argumentative Essays**

This repository contains the implementation of a Named Entity Recognition (NER) model for analyzing argumentative essays using transformer-based architectures. The goal of this project is to identify entities related to argument structure in student essays to evaluate the quality of arguments. 10.5281/zenodo.14829503

## **Features**

* Dataset: Feedback Prize \- Evaluating Student Writing (Kaggle) https://www.kaggle.com/competitions/feedback-prize-effectiveness/data

* Model: google/bigbird-roberta-base (Transformer-based)

* Libraries: PyTorch, Transformers (Hugging Face), Scikit-learn

* Key functionalities: Data preprocessing, NER model training, evaluation, and inference.

---

## **Installation**

To use the code, follow these steps:

### **Prerequisites**

1. **Python**: Ensure you have Python 3.9 or higher installed.

**Libraries**: Install the required libraries using the following command:

`pip install torch transformers scikit-learn pandas numpy tqdm`

2. **GPU Support**: For efficient training, ensure that your system has a compatible GPU and CUDA installed.

---

### **Steps for Implementation**

**Clone the Repository**  
Clone this repository to your local system:

`git clone https://github.com/username/ner-argument-essays.git`

`cd ner-argument-essays`

**Prepare the Dataset**  
Download the *Feedback Prize \- Evaluating Student Writing* dataset from Kaggle (link here) and place it in the `data/` directory.

**Data Preprocessing**  
Run the preprocessing script to clean and prepare the dataset for NER:

`python preprocess_data.py`

**Model Training**  
Train the NER model using the prepared dataset:

`python train_model.py --epochs 10 --batch_size 16 --learning_rate 5e-5`

Adjust hyperparameters (`epochs`, `batch_size`, `learning_rate`) as needed.

**Validation and Testing**  
Evaluate the model's performance on the validation dataset:

`python evaluate_model.py`

This will output metrics such as F1 score, precision, and recall.

**Inference**  
Use the trained model to perform inference on new, unlabeled data:

`python infer.py --input_file data/new_essays.txt --output_file results/predictions.json`

---

## **Directory Structure**

bash

CopyEdit

`ner-argument-essays/`

`│`

`├── data/`

`│   ├── train.csv          # Training dataset`

`│   ├── test.csv           # Testing dataset`

`│   └── new_essays.txt     # New essays for inference`

`│`

`├── models/`

`│   └── bigbird-roberta/   # Pre-trained transformer model`

`│`

`├── scripts/`

`│   ├── preprocess_data.py # Data preprocessing script`

`│   ├── train_model.py     # Model training script`

`│   ├── evaluate_model.py  # Model evaluation script`

`│   └── infer.py           # Inference script`

`│`

`└── README.md              # Project documentation`

---

## **Results**

The model achieves an F1 score of **XX.XX%**, precision of **XX.XX%**, and recall of **XX.XX%** on the validation dataset.

---

## **Limitations**

This implementation is optimized for a specific dataset and may require adjustments for other datasets or tasks. Additionally, GPU support is recommended for training, as training on a CPU may be time-consuming.

---

## **Contributing**

Contributions are welcome\! If you find issues or have suggestions, feel free to open an issue or submit a pull request.

---

