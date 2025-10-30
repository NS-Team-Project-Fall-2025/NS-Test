# Django RAG Chatbot

This is a complete Django project that implements a RAG (Retrieval-Augmented Generation) chatbot.

This version is configured to be **100% offline**.

* **Embeddings:** Uses a free, local **Hugging Face** model (`all-MiniLM-L6-v2`) to create and retrieve document embeddings.

* **Generation:** Uses a free, local **Ollama** model (`mistral`) to generate answers and MCQs.

## Step 1: System Setup (Ollama)

Before running the project, you must install Ollama and download the model.

1.  **Install Ollama:** Go to <https://ollama.com/> and download the application for your OS (e.g., macOS).

2.  **Run Ollama:** Make sure the Ollama application is running in the background.

3.  **Pull the Model:** Open your terminal and run this command to download the `mistral` model. This is a one-time download and may take a few minutes.

    ```
    ollama pull mistral
    ```

## Step 2: Project Setup

1.  **Clone/Unzip Project:** Make sure you have all the project files.

2.  **Create Virtual Environment:**

    ```
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:** Make sure your `requirements.txt` file is up-to-date (it should include `langchain-ollama`) and run:

    ```
    pip install -r requirements.txt
    ```

4.  **Add Documents:** Create a folder named `docs` in the main project directory. Place your PDF files inside this `docs` folder.

## Step 3: Run the App

1.  **Ingest Documents (One-time only):** Run the ingestion command. This will read your `docs`, create local embeddings, and save the `faiss_index` file.

    ```
    python manage.py ingest_docs
    ```

2.  **Run the Server:**

    ```
    python manage.py runserver
    ```

3.  **Open in Browser:** Go to `http://127.0.0.1:8000/` to use your app.