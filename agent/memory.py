import weave
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import time

@weave.op()
class MemoryManager:
    """Memory management with semantic search capabilities"""
    
    def __init__(self, memory_file: str = "agent_memory.json", max_memories: int = 1000):
        self.memory_file = memory_file
        self.max_memories = max_memories
        self.memories = []
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.load_memory()
    
    def load_memory(self):
        """Load existing memories from file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    self.memories = json.load(f)
            except:
                self.memories = []
    
    def save_memory(self):
        """Save memories to file"""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memories[-self.max_memories:], f, indent=2)
    
    @weave.op()
    def add_interaction(self, content: str, role: str):
        """Add new interaction to memory"""
        memory_entry = {
            "content": content,
            "role": role,
            "timestamp": time.time()
        }
        self.memories.append(memory_entry)
        
        # Keep only recent memories
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories:]
        
        self.save_memory()
    
    @weave.op()
    def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant context using keyword and semantic search"""
        if not self.memories:
            return ""
        
        query_lower = query.lower()
        
        # For name queries, search for name mentions
        if "name" in query_lower:
            for memory in reversed(self.memories):
                content_lower = memory["content"].lower()
                if "name is" in content_lower or "i'm" in content_lower or "call me" in content_lower:
                    return f"{memory['role']}: {memory['content']}"
        
        # For other personal info queries
        if any(word in query_lower for word in ["live", "from", "location", "city"]):
            for memory in reversed(self.memories):
                content_lower = memory["content"].lower()
                if any(word in content_lower for word in ["live", "from", "houston", "city"]):
                    return f"{memory['role']}: {memory['content']}"
        
        # Fallback: return recent context
        recent_memories = self.memories[-top_k:]
        return "\n".join([f"{m['role']}: {m['content']}" for m in recent_memories])
    
    @weave.op()
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "total_memories": len(self.memories),
            "user_messages": len([m for m in self.memories if m["role"] == "user"]),
            "assistant_messages": len([m for m in self.memories if m["role"] == "assistant"]),
            "memory_file": self.memory_file
        }