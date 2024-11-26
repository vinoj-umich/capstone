# AutoSensAI Chatbot

## Overview

AutoSensAI Chatbot leverages Retrieval-Augmented Generation (RAG) architecture with pre-trained Large Language Models (LLMs). The system incorporates open-source tools and NVIDIA GPUs to create a chatbot that can interact with domain-specific datasets such as PDFs and knowledge bases. The goal is to develop a robust chatbot for tasks such as document comprehension, customer support, and query answering.

## Getting Started

### Prerequisites

1. Clone the repository:
```bash
git clone https://github.com/vinoj-umich/capstone.git
```

2. Install the required Python package dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure Python 3.11 is installed and configured. The setup has been tested on Windows 11 with an NVIDIA RTX 4090 (CUDA 12.1).

4. A CUDA-compatible GPU with at least 5GB of VRAM is recommended to run the pipeline locally. Alternatively, use Google Colab for cloud-based execution.

5. Agree to the terms and conditions for accessing Gemma LLM models via the Hugging Face Hub.

### Project Structure

- `/Dataset`: Stores the ingested and preprocessed datasets (e.g., chunked PDFs).
- `/LLM`: Contains pre-trained large language models used in the project.
- `/Retrievers`: Embedding models for similarity searches and retrieval tasks.
- `/Notebook`: Jupyter notebooks for pipeline setup and experimentation.
- `/Interface`: Streamlit-based chatbot for answering user queries with a local host setup. The chatbot supports conversational features like “chat with PDF.”

### Workflow

1. Preprocess datasets using `DataPreProcess.ipynb` to create train and test datasets from raw knowledge articles.
2. Experiment with prompt engineering and model evaluation using notebooks under `/LLM`.
3. Experiment and benchmark retrievers using notebooks under `/Retrievers`.
4. Execute `GenerateAnswers.ipynb` to create synthetic ground truth answers for training purposes.
5. Evaluate the complete RAG pipeline with `rag_evaluation.ipynb` using the test dataset.

### Features

- **PDF Ingestion to Chatbot:** End-to-end pipeline for PDF ingestion and chatbot querying, running locally on NVIDIA GPUs.
- **Flexibility:** Use Google Colab for cloud-based execution.
- **Extensions:** Option to add external tools for speedup and performance enhancement, such as Flash Attention.

### Notes

- For issues during installation or execution, please refer to the Hugging Face documentation or leave an issue on the GitHub repository.
- Pre-trained models, such as Gemma-7B, are used for generation, and Flash Attention 2 can be optionally installed for faster inference.

### Data Access

The chatbot uses knowledge articles and documents provided by the user. The pipeline supports ingestion of open-source materials such as Wikipedia articles or proprietary datasets for retrieval and query answering.

### Contributors

1. Vinoj Bethellu
2. Kiran Irde
3. Shiva Vupputuri

### Project Report

Comprehensive documentation and video tutorials are available in the repository and linked to the README file.