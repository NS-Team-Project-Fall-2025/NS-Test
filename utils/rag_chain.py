from typing import List, Dict, Any, Optional
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
        """Generate the prompt for the LLM (basic single-turn)."""
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context. 
        Use the following context to answer the user's question. If the answer cannot be found in the context, 
        say so clearly.

Context:
{context}

Question: {query}

Answer: Please provide a comprehensive answer based on the context above. If the information is not 
available in the context, please state that clearly."""
        
        return prompt
    
    def _extract_last_user_questions(self, chat_history: Optional[List[Dict]]) -> List[str]:
        """Extract user messages from history, return last two before the most recent user message.
        Assumes chat_history includes the current user message at the end.
        """
        if not chat_history:
            return []
        user_msgs = [m.get("content", "") for m in chat_history if isinstance(m, dict) and m.get("role") == "user"]
        if not user_msgs:
            return []
        # The last entry in user_msgs is the current question; take the two before it
        if len(user_msgs) <= 1:
            return []
        prior = user_msgs[:-1]
        return prior[-2:]
    
    def _rewrite_retrieval_query(self, last_two: List[str], current: str) -> str:
        """Try to rewrite the three questions into a single standalone retrieval query.
        Falls back to simple concatenation if the model call fails.
        """
        # Fallback first (always available)
        concatenated = " \n".join([q for q in last_two if q] + [current])
        try:
            instruction = (
                "Rewrite the following conversation of user questions into ONE standalone search query. "
                "Resolve pronouns and references so that it is fully self-contained. "
                "Focus on the current question while keeping useful specifics from the prior two.\n\n"
                f"Previous Q1: {last_two[-2] if len(last_two) == 2 else ''}\n"
                f"Previous Q2: {last_two[-1] if last_two else ''}\n"
                f"Current Q: {current}\n\n"
                "Standalone search query:"
            )
            resp = self.model.invoke(instruction)
            if isinstance(resp, str) and len(resp.strip()) >= 5:
                return resp.strip()
        except Exception:
            pass
        return concatenated.strip()
    
    def _format_recent_turns(self, chat_history: Optional[List[Dict]], max_messages: int = 6) -> str:
        """Format the last few conversation messages for continuity."""
        if not chat_history:
            return "(no prior conversation)"
        recent = chat_history[-max_messages:]
        lines = []
        for m in recent:
            role = m.get("role", "assistant") if isinstance(m, dict) else "assistant"
            content = m.get("content", "") if isinstance(m, dict) else str(m)
            role_name = "User" if role == "user" else "Assistant"
            lines.append(f"{role_name}: {content}")
        return "\n".join(lines)
    
    def _build_final_prompt(self, latest_question: str, context: str, recent_turns: str) -> str:
        """Build the final multi-part prompt for the LLM."""
        return (
            "You are a helpful AI assistant. Ground your answer primarily in the provided knowledge base snippets. "
            "If the answer is not covered there, say you don't have enough information. Be concise and precise.\n\n"
            "Recent conversation:\n"
            f"{recent_turns}\n\n"
            "Retrieved knowledge base snippets:\n"
            f"{context}\n\n"
            "User's latest question (answer this):\n"
            f"{latest_question}\n\n"
            "Answer:"
        )
    
    def generate_answer(self, query: str, k: int = 4) -> Dict[str, Any]:
        """Generate answer using RAG pipeline (single-turn)."""
        try:
            # Step 1: Retrieve relevant documents
            documents = self.retrieve_documents(query, k=k)
            
            if not documents:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "context": "",
                    "query": query
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
        """Conversational RAG with history-aware retrieval and grounded answering."""
        try:
            chat_history = chat_history or []
            # Extract last two prior user questions and rewrite retrieval query
            last_two = self._extract_last_user_questions(chat_history)
            combined_retrieval_query = self._rewrite_retrieval_query(last_two, query)
            
            # Retrieve using combined query
            documents = self.retrieve_documents(combined_retrieval_query, k=k)
            
            # Format context
            context = self.format_context(documents)
            
            # Build final prompt including recent turns and the latest question
            recent_turns = self._format_recent_turns(chat_history, max_messages=6)
            final_prompt = self._build_final_prompt(query, context, recent_turns)
            
            # Invoke LLM
            answer = self.model.invoke(final_prompt)
            
            # Sources
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
                "query": query,
                "retrieval_query": combined_retrieval_query
            }
        except Exception as e:
            return {
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "sources": [],
                "context": "",
                "query": query
            }