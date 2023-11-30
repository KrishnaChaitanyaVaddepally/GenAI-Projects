# Student Score Prediction

## Overview

This project predicts students' scores using a machine learning model based on both categorical and numerical features. The workflow covers data exploration, preprocessing, model training, evaluation, and deployment.

## Project Structure

The project is organized as follows:

- `data/`: Contains datasets used for training and testing.
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, and model development.
- `src/`: Python source code for model training and evaluation.
- `model/`: Saved models and model-related files.
- `requirements.txt`: List of dependencies required to run the project.

## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/KrishnaChaitanyaVaddepally/mlproject.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd mlproject
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data

The dataset used in this project is available in the `notebook/data/` directory. Ensure the data is correctly loaded and preprocess it as needed before training the model.

## Notebooks

Explore the Jupyter notebooks in the `notebook/1 . EDA STUDENT PERFORMANCE .ipynb/` directory for detailed steps on data exploration, feature engineering. Model development is implemented in `notebook/2. MODEL TRAINING.ipynb/` . These notebooks serve as a guide through the project workflow.

## Training the Model

The machine learning model is implemented in Python using scikit-learn and CatBoost. To train the model, run:
```bash
python src/components/data_ingestion.py
```

## Evaluate
Evaluate the model's performance by running:

## Contributing

Feel free to contribute to this project by forking the repository and submitting pull requests. Your contributions are highly valued!




