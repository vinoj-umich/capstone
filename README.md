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

- `/chroma_db_dir`: Stores the ingested and preprocessed datasets (e.g., chunked PDFs).
<!-- - `/LLM`: Contains pre-trained large language models used in the project. -->
- `/Manuals` : Raw Data manuals to be processed and converted to chunks and embeddings
- `/exploration` : Evaluation of different models and comparing with ground truth 
- `/`: Embedding models for similarity searches and retrieval tasks. Streamlit-based chatbot for answering user queries with a local host setup. 

### Workflow

1. Preprocess datasets using `document_processor.ipynb` to create train and test datasets from raw pdf manuals.
2. Experiment with prompt engineering and model evaluation using `rag_evaluation.ipynb`
3. Experiment and benchmark retrievers using notebooks  `info_retrieval_eval.ipynb`.
4. document_processor.ipynb contains class to create synthetic ground truth answers for training purposes.
<!-- 5. Evaluate the complete RAG pipeline with `rag_evaluation.ipynb` using the test dataset. -->
5. chatbot.py contains streamlit and model usage for chatbot interaction.

### Features

- **PDF Ingestion to Chatbot:** End-to-end pipeline for PDF ingestion and chatbot querying, running locally on NVIDIA GPUs.
- **Flexibility:** Use Google Colab for cloud-based execution.
- **Extensions:** Option to add external tools for speedup and performance enhancement, such as Flash Attention.

### Notes

- For issues during installation or execution, please refer to the Hugging Face documentation or leave an issue on the GitHub repository.
- Pre-trained models, such as Gemma-7B, are used for generation, and Flash Attention 2 can be optionally installed for faster inference.

### Data Access

The chatbot uses pre-processed pdf manuals and user interacts with chatbot with answers. 

### Contributors

1. Vinoj Bethelli: Architect and Lead developer.
2. Kiran Irde: Streamlit coding and documentation.
3. Siva Vupputuri: Project timelines, exploration and code development.

### Project Report

Comprehensive documentation and video tutorials will be available in the repository and linked to the README file.