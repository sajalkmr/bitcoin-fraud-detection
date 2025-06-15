# Bitcoin Fraud Detection using Machine Learning

This project implements various machine learning models, including classical approaches and Graph Convolutional Network (GCN) with Multi-Layer Perceptron (MLP), for detecting fraudulent transactions in the Bitcoin network. The project utilizes the Elliptic dataset, which provides a rich set of features for transaction analysis.

## Project Structure

The repository is organized as follows:

-   `app.py`: Streamlit web application for interactive model training and data analysis.
-   `classical_ml.py`: Contains implementations of traditional machine learning models (e.g., Logistic Regression, Random Forest, SVM) for fraud detection.
-   `data_preprocessing.py`: Scripts for cleaning, transforming, and preparing the raw transaction data for model training.
-   `gcn_mlp.py`: Implements the Graph Convolutional Network (GCN) and Multi-Layer Perceptron (MLP) hybrid model for leveraging graph-structured data.
-   `models.py`: Defines the architecture and training logic for various machine learning models used in the project.
-   `read_data.py`: Handles the loading and initial parsing of the Elliptic dataset.
-   `requirements.txt`: Lists all necessary Python dependencies for the project.
-   `utils.py`: Contains utility functions and helper scripts used across different modules.
-   `visualization.py`: Scripts for generating plots and visualizations of data, model performance, and transaction patterns.
-   `dataset/`: Directory containing the Elliptic transaction dataset files.
    -   `elliptic_txs_classes.csv`: Transaction labels (licit/illicit).
    -   `elliptic_txs_edgelist.csv`: Graph structure representing transactions.
    -   `elliptic_txs_features.csv`: Features for each transaction.
-   `figures/`: Directory to store generated plots, charts, and system diagrams.
    -   `sys_model.png`: Example system model diagram.

## Dataset

This project uses the [Elliptic Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set). It is a dataset of Bitcoin transactions, where transactions are classified as licit or illicit. The dataset includes transaction features and a graph structure representing connections between transactions.

## Installation

To set up the project environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sajalkmr/bitcoin-fraud-detection.git
    cd bitcoin-fraud-detection
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    -   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

After installation, you can run the various scripts to preprocess data, train models, and visualize results.

-   **Data Preprocessing:**
    ```bash
    python data_preprocessing.py
    ```
-   **Train Classical ML Models:**
    ```bash
    python classical_ml.py
    ```
-   **Train GCN-MLP Model:**
    ```bash
    python gcn_mlp.py
    ```
-   **Run Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
-   **Generate Visualizations:**
    ```bash
    python visualization.py
    ```

Refer to individual script files for more specific usage details and command-line arguments.


## License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).
