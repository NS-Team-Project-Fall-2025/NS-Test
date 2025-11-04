import os
import json
from django.shortcuts import render
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache
# --- LangChain Imports ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import PromptTemplate

# --- Constants ---
VECTORSTORE_FILE = "faiss_index"

# --- Globals ---
RETRIEVER = None

def load_retriever():
    """
    Loads the vector store using local HuggingFace embeddings
    and initializes just the retriever.
    """
    global RETRIEVER

    if not os.path.exists(VECTORSTORE_FILE):
        print("Vector store file not found. Run 'python manage.py ingest_docs'")
        return

    try:
        # --- Use the *same* free, local model to load the store ---
        model_name = "all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # Load the vector store
        vectorstore = FAISS.load_local(
            VECTORSTORE_FILE, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # --- Create a retriever from the vector store ---
        RETRIEVER = vectorstore.as_retriever()
        
        print("--- Document Retriever (Local Embeddings) loaded successfully. ---")
    
    except Exception as e:
        print(f"Error loading retriever: {e}")

# --- Views ---

def index(request):
    """
    Render the main chat page.
    """
    if RETRIEVER is None:
        load_retriever()
    return render(request, 'chat/index.html')

@csrf_exempt
def chat_api(request: HttpRequest):
    """
    API endpoint to handle user chat messages.
    This endpoint only performs RETRIEVAL, not generation.
    It returns the raw text chunks.
    """
    if RETRIEVER is None:
        load_retriever() # Try loading again
        if RETRIEVER is None:
            return JsonResponse({
                'answer': "Error: Vector store file 'faiss_index' not found.\n"
                          "Please run the ingestion command: python manage.py ingest_docs"
            }, status=500)

    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    
    data = json.loads(request.body)
    query = data.get('message')
    
    if not query:
        return JsonResponse({'error': 'No message provided'}, status=400)

    try:
        # --- Get relevant documents (chunks) from the retriever ---
        relevant_docs = RETRIEVER.get_relevant_documents(query)
        
        if not relevant_docs:
            return JsonResponse({'answer': "No relevant information found in your documents.", 'sources': []})

        # --- Format the retrieved chunks as the answer ---
        formatted_answer = "Found the following relevant information in your documents:\n\n"
        formatted_answer += "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

        sources = [
            {
                'page': doc.metadata.get('page', 'N/A'),
                'source': doc.metadata.get('source', 'N/A').split('/')[-1]
            } 
            for doc in relevant_docs
        ]
        
        return JsonResponse({'answer': formatted_answer, 'sources': sources})
    
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return JsonResponse({'answer': f"An error occurred during retrieval: {e}"}, status=500)


@csrf_exempt
def mcq_api(request: HttpRequest):
    """
    API endpoint for MCQ generation.
    This feature is NOW 100% OFFLINE using Ollama.
    """
    if RETRIEVER is None: # Use the loaded retriever
        load_retriever()
        if RETRIEVER is None:
            return JsonResponse({
                'error': "Vector store not loaded. Run 'python manage.py ingest_docs'."
            }, status=500)

    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    
    data = json.loads(request.body)
    topic = data.get('topic', 'a key concept from the document')
    
    try:
        # 1. Retrieve relevant chunks (using local embeddings)
        relevant_chunks = RETRIEVER.get_relevant_documents(topic)
        
        if not relevant_chunks:
            # --- FEATURE 3: Topic Scoping ---
            return JsonResponse({
                'error': f"No questions to ask: could not find relevant information on '{topic}'.",
                'no_questions': True
            }, status=404)

        # --- CONTEXT FIX: Build context with source indexes ---
        context_parts = []
        page_map = {} # To store the page number for each chunk index
        for i, chunk in enumerate(relevant_chunks):
            page = chunk.metadata.get('page', 'N/A')
            # Add the source index to the context for the LLM
            context_parts.append(f"[Source {i}]\n{chunk.page_content}")
            page_map[i] = page # Map the index to its page number
        
        context_text = "\n\n---\n\n".join(context_parts)

        # --- 100% OFFLINE GENERATION WITH OLLAMA ---
        
        # 2. Initialize the local Ollama LLM
        # --- MODEL UPGRADE: Using 'mistral' for better quality ---
        # This is a more powerful model than 'phi3:mini'
        llm = ChatOllama(model="mistral", temperature=0.3)

        # 3. --- PROMPT FIX: Update prompt to ask for source_index ---
        mcq_prompt_template = """
        Based *only* on the context provided below, generate one (1) multiple-choice question (MCQ).
        The context is given as a list of sources, each with an index like [Source 0], [Source 1], etc.
        The question should be about "{topic}".
        
        Provide:
        1. The "question"
        2. Four "options" as a list of strings
        3. The "correct_answer" (must be one of the four options)
        4. The "source_index" (the index number, e.g., 0, 1, 2... of the ONE source chunk you used to create the question)
        
        Format your response *only* as a single valid JSON object.
        
        Example format:
        {{
            "question": "What is the capital of France?",
            "options": ["London", "Berlin", "Paris", "Madrid"],
            "correct_answer": "Paris",
            "source_index": 1
        }}

        --- CONTEXT ---
        {context}
        --- END CONTEXT ---

        JSON Response:
        """
        
        # 4. Create the generation chain
        prompt = PromptTemplate(
            template=mcq_prompt_template,
            input_variables=["context", "topic"]
        )
        
        llm_chain = prompt | llm
        
        # 5. Generate the response
        response_text = llm_chain.invoke({
            "context": context_text, 
            "topic": topic
        }).content
        
        # 6. Clean and parse the JSON response
        if response_text.strip().startswith("```json"):
            response_text = response_text.strip()[7:-3].strip()
        
        mcq_data = json.loads(response_text)
        
        # --- PAGE NUMBER FIX: Use the source_index from the LLM ---
        source_index = mcq_data.get('source_index', 0) # Get the index from the LLM
        
        # Look up the correct page number from the map we created
        mcq_data['page'] = page_map.get(source_index, 'N/A') 
        
        return JsonResponse(mcq_data)

    except json.JSONDecodeError:
        print(f"Failed to parse JSON from LLM: {response_text}")
        return JsonResponse({'error': 'Failed to generate MCQ (JSON parse error).'}, status=500)
    except Exception as e:
        print(f"Error during MCQ generation: {e}")
        return JsonResponse({'error': f"An error occurred: {e}"}, status=500)

