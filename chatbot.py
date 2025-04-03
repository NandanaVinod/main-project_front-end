from main import MilvusVectorDB
from openai import OpenAI
import logging
from typing import List, Dict, Tuple
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hugging Face token - replace with your token if needed
HF_TOKEN = os.getenv("HF_TOKEN")

class ChatBot:
    def __init__(self):
        self.conversation_history: List[Dict] = []
        self.max_history = 5
        self.vector_db = MilvusVectorDB()
        self.model_name = "Qwen/Qwen2.5-72B-Instruct"
        
        # Initialize client with custom URL
        self.client = OpenAI(
            base_url="https://api-inference.huggingface.co/v1/",
            api_key=HF_TOKEN
        )
        logger.info(f"ChatBot initialized with model: {self.model_name}")
        self.persona = """Role: Your name is AIVO (Advanced Intelligent Virtual Orator). You are a knowledgeable professor at KTU University.

        Objective: Keep your responses strictly focused on the question asked, providing clear, concise explanations using only the relevant context.

        Guidelines:
        - If any part of the provided input contains image content, ignore it and process the remaining details.
        - Ensure responses are direct, non-conversational, and contain no additional or unrelated information.
        - If the exact answer is not found in the context provided, state explicitly: "Insufficient information."
        - Process only text-based information from the reference materials.
        - Maintain a professional, academic tone throughout responses.
        - Focus solely on factual information from the provided context.
        - dont print the thinking part just the answer. """

    def truncate_text(self, text: str, max_chars: int = 500) -> str:
        """Truncate text to maximum character length at word boundary."""
        return text[:max_chars] + "..." if len(text) > max_chars else text

    def get_relevant_context(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve relevant context from Milvus"""
        logger.info(f"Retrieving context for query: {query}")
        return self.vector_db.search(query, top_k=top_k)

    def get_response(self, query: str) -> str:
        contexts = self.get_relevant_context(query, top_k=2)
        context_texts = [text for text, _ in contexts]
        context_text = "\n---\n".join([self.truncate_text(ctx) for ctx in context_texts]) if context_texts else "No relevant context available."

        system_message = f"""{self.persona}

        Reference Material:
        {context_text}

        Task: Provide a focused explanation addressing only the specific question, using the Reference Material to support your response. Adhere to the guidelines specified in the Role.

        User's Query:
                """

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
                top_p=0.95,
            )
            
            response = completion.choices[0].message.content.strip() if completion.choices else "No response generated."
            
            print("\nProf. AIVO:", response, "\n")
            
            # Update conversation history
            self.conversation_history.append({
                "user": query,
                "assistant": response,
                "timestamp": datetime.now().isoformat(),
                "contexts": context_texts
            })
            
            if len(self.conversation_history) > self.max_history:
                self.conversation_history.pop(0)
                
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered a technical issue. Please try your question again."

# def main():
#     chatbot = ChatBot()
#     print("Welcome to KTU Virtual Professor Assistant - AIVO")
#     print("Type 'quit' to exit the session.")
#     print("You can ask questions about the course materials.")
    
#     while True:
#         try:
#             query = input("\nYou: ").strip()
            
#             if query.lower() in ['quit', 'exit', 'bye']:
#                 print("\nGoodbye!")
#                 break
                
#             if not query:
#                 continue
                
#             chatbot.get_response(query)
            
#         except KeyboardInterrupt:
#             print("\nGoodbye!")
#             break
#         except Exception as e:
#             print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()