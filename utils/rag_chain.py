from typing import List, Dict, Any
from langchain.schema import Document
from langchain_community.llms import Ollama
from utils.vector_store import VectorStore
from config import Config

class RAGChain:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.setup_ollama()
    
    def setup_ollama(self):
        """Setup Ollama LLM."""
        self.model = Ollama(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_MODEL,
            temperature=Config.OLLAMA_TEMPERATURE
        )
    
    def retrieve_documents(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents for the query."""
        return self.vector_store.similarity_search(query, k=k)
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('filename', 'Unknown source')
            content = doc.page_content.strip()
            context_parts.append(f"Document {i} (Source: {source}):\n{content}")
        
        return "\n\n".join(context_parts)
    
    def generate_prompt(self, query: str, context: str) -> str:
        """Generate the prompt for the LLM."""
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context. 
        Use the following context to answer the user's question. If the answer cannot be found in the context, 
        say so clearly.

Context:
{context}

Question: {query}

Answer: Please provide a comprehensive answer based on the context above. If the information is not 
available in the context, please state that clearly."""
        
        return prompt
    
    def generate_answer(self, query: str, k: int = 4) -> Dict[str, Any]:
        """Generate answer using RAG pipeline."""
        try:
            # Step 1: Retrieve relevant documents
            documents = self.retrieve_documents(query, k=k)
            
            if not documents:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "context": ""
                }
            
            # Step 2: Format context
            context = self.format_context(documents)
            
            # Step 3: Generate prompt
            prompt = self.generate_prompt(query, context)
            
            # Step 4: Get response from Ollama
            response = self.model.invoke(prompt)
            answer = response
            
            # Step 5: Extract sources
            sources = []
            for doc in documents:
                source_info = {
                    "filename": doc.metadata.get('filename', 'Unknown'),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "context": context,
                "query": query
            }
            
        except Exception as e:
            return {
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "sources": [],
                "context": "",
                "query": query
            }
    
    def chat_with_context(self, query: str, chat_history: List[Dict] = None, k: int = 4) -> Dict[str, Any]:
        """Enhanced chat function with conversation history."""
        # For now, we'll implement basic RAG. 
        # You can extend this to include chat history in the prompt
        return self.generate_answer(query, k=k)