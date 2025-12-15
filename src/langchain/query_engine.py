# Week 2 Day 11-12: Natural Language Query Engine (FIXED)
# File: src/langchain/query_engine.py

import json
import re
import sys
import os
from typing import Dict, List, Optional
from datetime import datetime

# Fix imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from gpt4all import GPT4All

# Import vector store from same directory
from vector_store import SurveillanceVectorStore


class SurveillanceQueryEngine:
    """
    Natural language query engine for surveillance system
    Understands user questions and generates intelligent responses
    """
    
    def __init__(self, vector_store: SurveillanceVectorStore, 
                 model_path: str = None):
        """
        Initialize query engine
        
        Args:
            vector_store: Initialized vector store
            model_path: Path to GPT4All model (will download if not exists)
        """
        self.vector_store = vector_store
        
        # Initialize GPT4All for response generation
        print("🔄 Loading language model...")
        self.llm = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")
        print("✅ Language model loaded!")
        
    def parse_query(self, user_query: str) -> Dict:
        """
        Parse user query to extract intent and parameters
        
        Args:
            user_query: Natural language query from user
            
        Returns:
            Dictionary with parsed query components
        """
        query_lower = user_query.lower()
        
        parsed = {
            'original_query': user_query,
            'intent': 'general_search',
            'filters': {},
            'time_range': None,
            'object_type': None,
            'color': None,
            'behavior': None
        }
        
        # Extract time references
        time_patterns = [
            (r'between (\d+):(\d+) (?:and|to) (\d+):(\d+)', 'time_range_hhmm'),
            (r'between (\d+)-(\d+) (?:pm|am)', 'time_range_simple'),
            (r'at (\d+):(\d+)', 'specific_time'),
            (r'after (\d+):(\d+)', 'after_time'),
            (r'before (\d+):(\d+)', 'before_time'),
        ]
        
        for pattern, time_type in time_patterns:
            match = re.search(pattern, query_lower)
            if match:
                parsed['time_range'] = (time_type, match.groups())
                parsed['intent'] = 'time_filtered_search'
                break
        
        # Extract colors
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 
                 'orange', 'purple', 'brown']
        for color in colors:
            if color in query_lower:
                parsed['color'] = color
                parsed['filters']['color'] = color
                break
        
        # Extract object types
        object_types = {
            'person': ['person', 'people', 'man', 'woman', 'human'],
            'car': ['car', 'vehicle', 'automobile'],
            'motorcycle': ['motorcycle', 'bike', 'motorbike'],
            'truck': ['truck'],
            'bus': ['bus'],
            'backpack': ['backpack', 'bag'],
            'handbag': ['handbag', 'purse'],
        }
        
        for obj_type, keywords in object_types.items():
            if any(keyword in query_lower for keyword in keywords):
                parsed['object_type'] = obj_type
                parsed['filters']['object_type'] = obj_type
                break
        
        # Extract behaviors
        behaviors = {
            'loitering': ['loiter', 'standing still', 'waiting'],
            'fast_movement': ['running', 'fast', 'quickly', 'rushing'],
            'erratic_movement': ['erratic', 'zigzag', 'unpredictable'],
            'suspicious': ['suspicious', 'unusual', 'strange']
        }
        
        for behavior_type, keywords in behaviors.items():
            if any(keyword in query_lower for keyword in keywords):
                parsed['behavior'] = behavior_type
                parsed['intent'] = 'behavior_search'
                break
        
        # Determine specific intents
        if any(word in query_lower for word in ['show', 'find', 'list']):
            if 'all' in query_lower or 'every' in query_lower:
                parsed['intent'] = 'list_all'
        
        if any(word in query_lower for word in ['how many', 'count']):
            parsed['intent'] = 'count_query'
        
        if any(word in query_lower for word in ['when', 'what time']):
            parsed['intent'] = 'temporal_query'
        
        return parsed
    
    def execute_query(self, user_query: str, max_results: int = 10) -> Dict:
        """
        Execute natural language query
        
        Args:
            user_query: User's natural language question
            max_results: Maximum number of results to return
            
        Returns:
            Query results with metadata
        """
        print(f"\n💬 User Query: '{user_query}'")
        
        # Parse query
        parsed = self.parse_query(user_query)
        print(f"🔍 Detected intent: {parsed['intent']}")
        if parsed['filters']:
            print(f"🏷️  Filters: {parsed['filters']}")
        
        # Execute search based on intent
        results = None
        
        if parsed['color']:
            results = self.vector_store.search_by_color(
                parsed['color'], 
                n_results=max_results
            )
        elif parsed['object_type']:
            results = self.vector_store.search_by_object_type(
                user_query, 
                parsed['object_type'], 
                n_results=max_results
            )
        else:
            results = self.vector_store.search(
                user_query, 
                n_results=max_results
            )
        
        return {
            'parsed_query': parsed,
            'results': results,
            'result_count': len(results['documents'])
        }
    
    def generate_response(self, user_query: str, query_results: Dict) -> str:
        """
        Generate natural language response using LLM
        
        Args:
            user_query: Original user query
            query_results: Results from vector search
            
        Returns:
            Natural language response
        """
        results = query_results['results']
        count = query_results['result_count']
        
        if count == 0:
            return "I couldn't find any events matching your query. Try rephrasing or being less specific."
        
        # Prepare context from search results
        context_items = []
        for i, (doc, meta) in enumerate(zip(results['documents'][:5], 
                                            results['metadatas'][:5])):
            context_items.append(f"{i+1}. {doc}")
        
        context = "\n".join(context_items)
        
        # Create prompt for LLM
        prompt = f"""You are an AI assistant for a surveillance system. Answer the user's question based on the search results.

User Question: {user_query}

Search Results Found: {count} events
Top Results:
{context}

Provide a clear, concise answer. Include specific details like:
- How many objects/events were found
- What types of objects (people, vehicles, etc.)
- Colors if mentioned
- Time information if available
- Any behaviors detected

Keep the response under 150 words and be specific.

Answer:"""
        
        print("🤖 Generating response...")
        
        # Generate response
        response = self.llm.generate(
            prompt,
            max_tokens=200,
            temp=0.7
        )
        
        return response.strip()
    
    def query(self, user_query: str, max_results: int = 10) -> str:
        """
        Complete query pipeline: parse -> search -> generate response
        
        Args:
            user_query: User's natural language question
            max_results: Maximum results to consider
            
        Returns:
            Natural language response
        """
        # Execute search
        query_results = self.execute_query(user_query, max_results)
        
        # Generate response
        response = self.generate_response(user_query, query_results)
        
        return response
    
    def get_query_examples(self) -> List[str]:
        """Get example queries users can try"""
        return [
            "Show me all people wearing red",
            "Find blue cars",
            "What vehicles appeared between 2-4 PM?",
            "Show suspicious behavior",
            "Find people with fast movement",
            "List all green objects",
            "How many people were detected?",
            "Show me everyone with a backpack",
            "Find erratic movement patterns",
            "What happened at 5 minutes?"
        ]


class InteractiveSurveillanceChat:
    """
    Interactive chat interface for surveillance queries
    """
    
    def __init__(self, query_engine: SurveillanceQueryEngine):
        self.query_engine = query_engine
        self.conversation_history = []
    
    def chat(self):
        """Start interactive chat session"""
        print("\n" + "=" * 70)
        print("🎯 SMART SURVEILLANCE - INTERACTIVE QUERY SYSTEM")
        print("=" * 70)
        print("\n💡 You can ask questions like:")
        examples = self.query_engine.get_query_examples()
        for i, example in enumerate(examples[:5], 1):
            print(f"   {i}. {example}")
        
        print("\n📝 Type 'examples' to see more query examples")
        print("📝 Type 'quit' or 'exit' to stop\n")
        
        while True:
            try:
                # Get user input
                user_input = input("🙋 You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Show examples
                if user_input.lower() == 'examples':
                    print("\n💡 Example Queries:")
                    for i, example in enumerate(self.query_engine.get_query_examples(), 1):
                        print(f"   {i}. {example}")
                    print()
                    continue
                
                # Process query
                print()  # Blank line for readability
                response = self.query_engine.query(user_input)
                
                print(f"\n🤖 Assistant: {response}\n")
                print("-" * 70 + "\n")
                
                # Save to history
                self.conversation_history.append({
                    'user': user_input,
                    'assistant': response,
                    'timestamp': datetime.now().isoformat()
                })
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again with a different query.\n")


# Main execution and testing
if __name__ == "__main__":
    print("=" * 70)
    print("🎯 NATURAL LANGUAGE QUERY ENGINE - SETUP & TESTING")
    print("=" * 70)
    
    # Initialize components
    print("\n📦 Initializing components...")
    vector_store = SurveillanceVectorStore('data/outputs/chroma_db')
    query_engine = SurveillanceQueryEngine(vector_store)
    
    print("\n✅ Query engine ready!")
    
    # Test queries
    print("\n" + "=" * 70)
    print("🧪 TESTING QUERIES")
    print("=" * 70)
    
    test_queries = [
        "Show me all people wearing blue",
        "Find vehicles or cars",
        "Show suspicious behavior",
        "Find people with fast movement",
        "What green objects were detected?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(test_queries)}")
        print('='*70)
        
        response = query_engine.query(query, max_results=5)
        print(f"\n🤖 Response:\n{response}")
    
    print("\n" + "=" * 70)
    print("✅ TESTING COMPLETE!")
    print("=" * 70)
    
    # Start interactive mode
    print("\n🎮 Starting interactive mode...")
    print("(You can quit anytime by typing 'quit')\n")
    
    chat = InteractiveSurveillanceChat(query_engine)
    chat.chat()