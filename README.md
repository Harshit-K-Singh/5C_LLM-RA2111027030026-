# LLM Fine-tuning for Impression Generation

## Overview

This project demonstrates proficiency in LLM (Large Language Model) fine-tuning, NLP (Natural Language Processing) techniques, and text analysis using a custom dataset. The goal is to fine-tune a pre-trained model to generate impressions based on medical reports provided in the dataset. The workflow involves model fine-tuning, evaluation, text preprocessing, and visualizing word similarities using embeddings.

---

## Project Structure

- **`src/`**: Contains the source code for fine-tuning the model, evaluating it, and performing text analysis.
- **`data/`**: Stores the dataset used for training and evaluation.
- **`outputs/`**: Contains generated outputs, including model evaluation metrics and visualizations.
- **`README.md`**: Project documentation (this file).
- **`requirements.txt`**: Contains the list of required Python packages.
- **`.gitignore`**: Specifies files and directories to be ignored by Git.
  
---

## Key Features

- Fine-tuning the `gemma-2b-it` model (or an alternative) to generate impressions based on report metadata.
- Evaluation metrics: Perplexity and ROUGE scores on a reserved evaluation dataset.
- Text analysis including stop-word removal, stemming, lemmatization, and embedding-based word similarity analysis.
- Visualization of the top 100 most similar word pairs.
- Optional: Interactive visualizations for exploring word similarities.

---

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- Git
- Hugging Face CLI (for model downloads)
- An active Hugging Face account

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/your-repository-name.git
    cd your-repository-name
    ```

2. **Set up a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    .\venv\Scripts\activate  # On Windows
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Authenticate with Hugging Face**:
    Log in to Hugging Face if you're using a gated model (e.g., `gemma-2b-it`):
    ```bash
    huggingface-cli login
    ```

---

## How to Use

1. **Fine-Tune the Model**:
    Fine-tune the model using the provided dataset:
    ```bash
    python src/main.py
    ```

2. **Evaluation**:
    The script will automatically evaluate the fine-tuned model on 30 reserved samples and provide metrics like **Perplexity** and **ROUGE** scores.

3. **Text Analysis**:
    Run text preprocessing and analysis to generate embeddings and identify the top 100 word pairs based on similarity.

4. **Visualization**:
    The code also generates visualizations of the top 100 similar word pairs. For an interactive visualization, explore the provided code in `src/visualization.py`.

---

## Model Evaluation Metrics

The evaluation process involves:
- **Perplexity**: Measures the uncertainty of the model in generating text.
- **ROUGE Score**: A metric that measures the overlap between generated text and the reference text (precision, recall, and F1-score).

---

## Text Analysis

1. **Preprocessing**:
    - Stop-word removal
    - Stemming and Lemmatization
2. **Embeddings**:
    The text is converted into embeddings for similarity analysis.
3. **Word Pairs**:
    Top 100 word pairs are identified based on their cosine similarity in the embedding space.

---

## Visualization

The top 100 similar word pairs are visualized using a scatter plot. An optional interactive visualization can be generated for exploring word relationships further.

---

## Project Files

- `src/main.py`: The main script to fine-tune and evaluate the model.
- `src/model_fine_tuning.py`: Script for loading and fine-tuning the model.
- `src/evaluation.py`: Handles evaluation metrics (perplexity, ROUGE).
- `src/text_analysis.py`: Contains text preprocessing and embedding generation code.
- `src/visualization.py`: Generates visualizations of word similarities.
- `requirements.txt`: Python dependencies required for the project.

---

## Future Improvements

- Incorporating a larger dataset to further improve the fine-tuning process.
- Enhancing the interactive visualization with more functionality.
- Adding more robust evaluation metrics, such as BLEU scores.

---

## Contributing

Contributions are welcome! If you would like to contribute, please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For any inquiries, please contact:
- **Name**: Harshit-K-Singh
- **Email**: singhgoluharshit494@gmail.com
