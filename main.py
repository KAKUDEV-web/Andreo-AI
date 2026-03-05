"""
Andreo AI Cloud Server - Render.com Fixed Version
Created by KAI
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Disable GPU (Render free tier has no GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
torch.set_num_threads(2)  # Limit CPU threads

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb

# ==================== CONFIG ====================
MODEL_NAME = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
MAX_MEMORY = int(os.getenv("MAX_MEMORY", "30"))

# ==================== PYDANTIC MODELS ====================
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 200

class MemoryRequest(BaseModel):
    session_id: str
    key: str
    value: str
    category: Optional[str] = "general"

class MemoryQuery(BaseModel):
    session_id: str
    query: str
    n_results: Optional[int] = 3

# ==================== ANDREO AI BRAIN ====================
class AndreoCloudBrain:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.embedder = None
        self.vector_dbs: Dict[str, chromadb.Collection] = {}
        self.sessions: Dict[str, List[Dict]] = {}
        self.start_time = time.time()
        self.load_model()
    
    def load_model(self):
        """Load AI model - CPU only for Render free tier"""
        print(f"🔄 Loading {MODEL_NAME} on CPU...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model on CPU (no quantization needed for small model)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # CPU uses float32
            low_cpu_mem_usage=True
        )
        
        # Load embedding model
        print("🔄 Loading embedder...")
        self.embedder = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device='cpu'
        )
        
        print("✅ Andreo AI Ready!")
    
    def get_vector_db(self, session_id: str):
        """Get or create vector database for session"""
        if session_id not in self.vector_dbs:
            client = chromadb.Client()
            self.vector_dbs[session_id] = client.create_collection(
                name=f"andrei_{session_id}",
                get_or_create=True
            )
        return self.vector_dbs[session_id]
    
    def build_prompt(self, message: str, session_id: str) -> str:
        """Build prompt with context"""
        # System
        prompt = "<|system|>\nYou are Andreo AI by KAI. Helpful, friendly, running on Render.com cloud.</s>\n"
        
        # Add memories
        try:
            db = self.get_vector_db(session_id)
            query_embedding = self.embedder.encode(message).tolist()
            results = db.query(
                query_embeddings=[query_embedding],
                n_results=2
            )
            if results["documents"] and results["documents"][0]:
                memories = " | ".join(results["documents"][0])
                prompt += f"<|memories|>\n{memories}</s>\n"
        except:
            pass
        
        # Add history
        history = self.sessions.get(session_id, [])
        for msg in history[-4:]:
            role = "user" if msg["role"] == "user" else "assistant"
            prompt += f"<|{role}|>\n{msg['content']}</s>\n"
        
        # Current message
        prompt += f"<|user|>\n{message}</s>\n<|assistant|>\n"
        
        return prompt
    
    def generate(self, request: ChatRequest) -> str:
        """Generate response"""
        prompt = self.build_prompt(request.message, request.session_id)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_beams=1,  # Faster
                early_stopping=True
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Save history
        if request.session_id not in self.sessions:
            self.sessions[request.session_id] = []
        
        self.sessions[request.session_id].append({
            "role": "user",
            "content": request.message
        })
        self.sessions[request.session_id].append({
            "role": "assistant",
            "content": response
        })
        
        # Trim
        if len(self.sessions[request.session_id]) > MAX_MEMORY * 2:
            self.sessions[request.session_id] = self.sessions[request.session_id][-MAX_MEMORY * 2:]
        
        return response
    
    def store_memory(self, request: MemoryRequest) -> bool:
        """Store in vector database"""
        try:
            db = self.get_vector_db(request.session_id)
            content = f"{request.key}: {request.value}"
            embedding = self.embedder.encode(content).tolist()
            
            db.add(
                ids=[f"{time.time()}"],
                embeddings=[embedding],
                documents=[content],
                metadatas=[{"category": request.category}]
            )
            return True
        except Exception as e:
            print(f"Memory error: {e}")
            return False
    
    def search_memories(self, request: MemoryQuery) -> List[str]:
        """Search stored memories"""
        try:
            db = self.get_vector_db(request.session_id)
            query_embedding = self.embedder.encode(request.query).tolist()
            
            results = db.query(
                query_embeddings=[query_embedding],
                n_results=request.n_results
            )
            
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "model": MODEL_NAME,
            "model_loaded": self.model is not None,
            "active_sessions": len(self.sessions),
            "uptime_seconds": int(time.time() - self.start_time)
        }

# Initialize
print("🚀 Starting Andreo AI...")
brain = AndreoCloudBrain()

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Andreo AI Cloud",
    description="Always-online AI by KAI - Render.com",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {
        "name": "Andreo AI",
        "company": "KAI",
        "status": "online",
        "url": "/docs for API docs"
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = brain.generate(request)
        return {
            "success": True,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remember")
async def remember(request: MemoryRequest):
    success = brain.store_memory(request)
    return {"success": success}

@app.post("/recall")
async def recall(request: MemoryQuery):
    memories = brain.search_memories(request)
    return {"memories": memories, "count": len(memories)}

@app.get("/status")
async def status():
    return brain.get_stats()

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
            
