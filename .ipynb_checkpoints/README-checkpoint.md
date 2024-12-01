# AutoSensAI Chatbot

## Overview

AutoSensAI Chatbot is a state-of-the-art, Retrieval-Augmented Generation (RAG) system that leverages pre-trained Large Language Models (LLMs) to interact with domain-specific datasets. The system combines open-source tools and NVIDIA GPUs to create a highly efficient and interactive chatbot capable of handling tasks like document comprehension, customer support, and query answering.

The chatbot utilizes Retrieval-Augmented Generation (RAG) for providing precise, context-aware answers by retrieving relevant information from a set of documents and generating human-like responses.

## Getting Started

### Prerequisites

Before running the AutoSensAI Chatbot, ensure the following prerequisites are met:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vinoj-umich/capstone.git
   ```

2. **Install the required Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure Python 3.11 is installed**. The system has been tested on Windows 11 with an NVIDIA RTX 4090 (CUDA 12.1). It is recommended to use Python 3.11 or above for optimal performance.

4. **Hardware Requirements**: 
   - A **CUDA-compatible GPU** with at least **5GB of VRAM** is required for running the pipeline locally.
   - Alternatively, you can use **Google Colab** or another cloud platform for running the pipeline if a local GPU is unavailable.

5. **Hugging Face Account**: 
   - You may need to authenticate with your **Hugging Face** account to access certain pre-trained models like **Gemma LLMs**. Make sure you agree to the necessary terms and conditions to access the models.

### Project Structure

The project directory is structured as follows:

- **`/chroma_db_dir`**: Contains the ingested and preprocessed datasets, such as chunked PDFs.
- **`/Manuals`**: Contains raw PDF data that will be processed and converted into embeddings and chunks.
- **`/pipeline`**: Includes python modules for custom classes based on scikir-learn,  additional details are explained below.
- **`/`**: Contains the chatbot logic, including Streamlit-based interaction, and the models used for retrieval and query answering.

### Custom Pipelines

The system uses several custom classes for building the data processing pipeline. These classes are designed to preprocess documents, generate embeddings, and facilitate model training and query answering. Below is a brief description of each class:

1. **`ChromaDBSaver`** (from `pipeline/chroma_db`): 
   - Responsible for saving and managing embeddings in a Chroma database. Chroma provides efficient retrieval of vector embeddings, which is critical for document-based question answering tasks. This class handles storing and retrieving document embeddings that are created during preprocessing.

2. **`PDFReader`** (from `pipeline/pdf_reader`):
   - This class is used to read and extract text from PDF documents. It handles the parsing of raw PDFs and transforms them into a format that can be further processed and chunked into smaller parts for indexing or retrieval.

3. **`TextFormatter`** (from `pipeline/text_proccessor`):
   - The `TextFormatter` class is responsible for cleaning and formatting text extracted from various sources, such as PDFs. It ensures that the text is structured correctly, removing unwanted characters and irrelevant information, making it suitable for further analysis.

4. **`SentenceChunkerWithSummarization`** (from `pipeline/chunk_proccessor`):
   - This class divides large texts into manageable chunks for efficient retrieval. It also includes summarization capabilities, which can be used to condense the content of large documents into concise segments, preserving essential information for answering questions.

5. **`QuestionAnswerGenerator`** (from `pipeline/question_generator`):
   - The `QuestionAnswerGenerator` class is responsible for generating potential questions based on the text chunks. These generated questions are important for training and evaluating the chatbot's ability to answer queries, as well as for augmenting the training dataset.

6. **`EmbeddingGenerator`** (from `pipeline/embedding_proccessor`):
   - The `EmbeddingGenerator` class creates vector embeddings from the text data using pre-trained language models. These embeddings allow the system to perform similarity searches efficiently, enabling the chatbot to retrieve the most relevant documents or document sections based on the user's query.

### Workflow

1. **Preprocess Data**: 
   - Use the `document_processor.ipynb` notebook to process raw PDF manuals. This will convert them into chunked datasets for retrieval and indexing. The custom classes in the pipeline, such as `PDFReader` and `SentenceChunkerWithSummarization`, play a crucial role in extracting and chunking the text from PDFs.

2. **Model Experimentation**: 
   - Experiment with different pre-trained models for query answering using `rag_evaluation.ipynb`. Evaluate the effectiveness of various models with respect to your data.

3. **Retriever Evaluation**: 
   - Benchmark different retrievers using the `info_retrieval_eval.ipynb` notebook. This allows you to evaluate retrieval performance for document-based question answering.

4. **Synthetic Ground Truth**: 
   - The `document_processor.ipynb` notebook also contains functionality for creating synthetic ground truth answers, which can be used for training purposes.

5. **Chatbot Integration**: 
   - The main chatbot logic is contained in `chatbot.py`. It uses Streamlit to create a user-friendly web interface and integrates with the models and retrievers to provide real-time query answering.

### Features

- **PDF Ingestion to Chatbot**: End-to-end pipeline for ingesting and processing PDFs, allowing the chatbot to query the documents.
- **Real-time Query Answering**: Users can interact with the chatbot and receive answers to domain-specific queries using a local or cloud-based LLM model.
- **Flexible and Extensible**: The system supports multiple models and retrieval methods and can be easily extended to add additional tools or optimizations.
- **Cloud Support**: If you don’t have access to a local GPU, you can use **Google Colab** for cloud-based execution, where the models can be run in the cloud.

### Notes

- **Flash Attention**: If you're working with large models and need faster inference, consider installing **Flash Attention 2**, which can significantly speed up processing time, especially when running on CUDA-enabled GPUs.
  
- **Model Compatibility**: The system supports a variety of pre-trained models, such as **Gemma-7B** and **Meta's Llama models**. These models can be accessed from the **Hugging Face Hub**.
  
- **Running Locally or in the Cloud**: The chatbot can be run either on your local machine (with an appropriate GPU) or on cloud-based platforms like **Google Colab**, which offers easy setup without requiring local GPU resources.

### Data Access

The chatbot uses **pre-processed PDFs** stored in the `/chroma_db_dir` directory. You interact with the chatbot by entering questions, and the chatbot retrieves relevant information from these documents to provide a response. 

The documents you use should be pre-processed into embeddings, chunked and indexed for efficient search and retrieval.

### Running the Chatbot

To run the chatbot locally, follow these steps:

1. Start the Streamlit application:
   ```bash
   streamlit run chatbot.py
   ```

2. A browser window will open, displaying the chatbot interface. You can now start asking questions and interact with the chatbot.

3. The chatbot will process your queries using the pre-trained models and retrieve relevant information from the ingested documents, providing real-time responses.

### Contributors

- **Vinoj Bethelli**: Architect and Lead Developer — Responsible for overall system design, model integration, and development.
- **Kiran Irde**: Streamlit UI Development and Documentation — Developed the chatbot interface and contributed to the documentation.
- **Siva Vupputuri**: Project Management and Development — Contributed to model evaluation and experimentation.

### Future Enhancements

- **Support for Multi-Document Queries**: Expand the chatbot's ability to handle queries that involve multiple documents simultaneously, offering more relevant answers.
- **Model Versioning**: Implement version control for models to make updates and upgrades easier to manage.
- **Faster Retrieval**: Further optimize the retrieval process using more advanced methods such as **FAISS** or **ElasticSearch**.
- **Enhanced User Interface**: Improve the UI for a more seamless and interactive user experience, including features like document previews or better query formatting.

### Troubleshooting

- For any issues during installation or usage, please refer to the official [Hugging Face documentation](https://huggingface.co/docs).
- If you encounter any specific errors or bugs, feel free to raise an issue in the [GitHub repository](https://github.com/vinoj-umich/capstone/issues).

---

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
