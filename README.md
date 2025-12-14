# ANIME RECOMMENDER SYSTEM

## General project description

This project implements an **anime recommendation system** that suggests relevant anime titles to users based on their preferences and historical interactions. The system is developed as an academic and experimental project, focusing on understanding and comparing different recommender system setups under varying data conditions.

The core idea of the project is to analyse how recommendation quality and system behavior change when:

* different **dataset sizes** are used,
* different **preprocessing strategies** are applied,
* different **training pipelines** are compared.

The project follows a clear and modular workflow:

* raw anime datasets are loaded from external sources,
* user-item interaction data and anime metadata are preprocessed,
* recommendation models are trained on different dataset versions,
* trained models or trained logic are used by an application layer to generate recommendations.

The repository separates data exploration, model training, and application usage to ensure clarity, reproducibility, and ease of experimentation.

---

## Data description

Two publicly available **Kaggle datasets** are used in this project. They represent different dataset versions and are intentionally used for different training stages.

### Anime Recommendations Database (original)

**Source:** [Kaggle Link](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

This dataset is **older and smaller dataset** and is mainly used in the experimental stage.

**Typical contents:**
* anime metadata (anime_id, name, genre, type, number of episodes),
* user-anime rating data,
* relatively smaller number of users and interactions compared to newer versions.

**Usage in the project:**
* training and experimentation in `train_v0.ipynb`,
* fast prototyping and debugging,
* validation of core recommendation logic.

### Anime Recommendation Database 2020

**Source:** [Kaggle Link](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)

This dataset is treated as the **newer and larger dataset** and is used for the main training pipeline.

**Typical contents:**
* extended anime metadata,
* larger and denser user-item interaction matrix,
* updated ratings and popularity information.

**Usage in the project:**
* main training experiments in `train_v1.ipynb`,
* evaluation of recommendation quality on a larger scale,
* generation of final results.

---

## How the data is used

**Recommended workflow for both datasets:**

* raw CSV files are downloaded from Kaggle,
* raw data is stored locally and not modified manually,
* all cleaning, filtering, and feature engineering is done inside the notebooks,
* each training notebook applies preprocessing logic appropriate for its dataset version.

### How to use Kaggle API to download the data

To download datasets programmatically using the Kaggle API:

1.  **Step 1.** Create a Kaggle account (if not already available).
2.  **Step 2.** Generate an API token:
    * go to Kaggle account settings,
    * create a new API token,
    * download the `kaggle.json` file.
3.  **Step 3.** Configure the API locally:
    * place your Username and Key into KAGGLE_USERNAME, KAGGLE_KEY credentials.
4.  **Step 4.** Install the Kaggle CLI:
    * install the `kaggle` package using `pip`.
5.  **Step 5.** Download datasets:
    * run the corresponding code cell.

After downloading, the notebooks can directly load the CSV files from the local data folder.

---

## Description of repository files

| File | Purpose | Key Notes |
| :--- | :--- | :--- |
| **`eda.ipynb`** | **Exploratory Data Analysis** of anime datasets. | Inspection of data structure, missing values, and distributions. Analytical only; does not produce final models. |
| **`train_v0.ipynb`** | Training on the **older and smaller dataset**. | Baseline recommendation experiments, rapid iteration. Primarily for experimentation. |
| **`train_v1.ipynb`** | Training on the **newer and larger dataset**. | Main and recommended training pipeline. Produces final experimental results. Use this to reproduce core findings. |
| **`quenrecommender-2.ipynb`** | Additional or **alternative recommender experiments**. | Exploration of non-core ideas. Running this notebook is optional. |
| **`app.py`** | **Application Entry Point**. | Serves as the main interface for the Streamlit web app. Solves "Cold Start" problem, allows dynamic profiling, uses LightFM model, and integrates Jikan API for covers. |
| **`rec.pdf`** | **Project report**. | Description of motivation, datasets, methodology, experiments, and conclusions. |

---

## How to start training

To start training the recommender system:

1.  Clone the repository and create a Python virtual environment.
2.  Install required Python dependencies (NumPy, pandas, scikit-learn, and other libraries used in the notebooks).
3.  Download both datasets and place them in the local data directory.
4.  Launch Jupyter Notebook or Jupyter Lab.
5.  Choose the training version:
    * `train_v0.ipynb` for experiments on the smaller dataset,
    * `train_v1.ipynb` for full training on the larger dataset (**recommended**).
6.  Run the notebook cells sequentially to preprocess data, train the model, and evaluate results.

---

## How to start the application

To run the recommender application:

1.  Ensure that the training has been completed and both `anime_lightfm_final.pkl` and `anime.csv` are present in the directory.
2.  Activate the Python environment used for training and ensure dependencies are installed.
3.  Run the application entry file using the Streamlit CLI:
    ```bash
    streamlit run app.py
    ```
4.  The application should automatically open in your default web browser (usually at `http://localhost:8501`).

Follow the on-screen prompts to verify your age and select anime to generate recommendations. The application demonstrates how trained recommender models can be used in practice using a modern interactive interface.

## External links:
- [Recommender v1](https://www.kaggle.com/code/hatitara/recommender-v2)
- [QUEN Recomender Pre](https://colab.research.google.com/drive/1Ea2lQaE9N_otfktAT2xrRJkijaiq8249?usp=sharing)