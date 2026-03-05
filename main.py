"""
Andreo AI Cloud Server
Render.com Deployment
Created by KAI
"""

import os
import time
import json
from datetime import datetime
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import chromadb

# ==================== CONFIG ====================
MODEL_NAME = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
USE_4BIT = os.getenv("USE_4BIT", "true").lower() == "true"
MAX_MEMORY = int(os.getenv("MAX_MEMORY", "50"))  # Number of messages to remember

# ==================== PYDANTIC MODELS ====================
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256

class MemoryRequest(BaseModel):
    session_id: str
    key: str
    value: str
    category: Optional[str] = "general"

class MemoryQuery(BaseModel):
    session_id: str
    query: str
    n_results: Optional[int] = 3

class StatusResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool
    active_sessions: int
    uptime_seconds: float

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
        """Load AI model with memory optimization"""
        print(f"🔄 Loading {MODEL_NAME}...")
        
        # Use smaller model for Render free tier (512MB RAM)
        if USE_4BIT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        else:
            # CPU-only for smaller models
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
        
        # Load embedding model for memory
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Andreo AI loaded!")
    
    def get_vector_db(self, session_id: str):
        """Get or create vector database for session"""
        if session_id not in self.vector_dbs:
            client = chromadb.Client()
            self.vector_dbs[session_id] = client.create_collection(
                name=f"andrei_mem_{session_id}",
                get_or_create=True
            )
        return self.vector_dbs[session_id]
    
    def build_prompt(self, message: str, session_id: str) -> str:
        """Build prompt with context and memories"""
        # System prompt
        prompt = "<|system|>\nYou are Andreo AI, created by KAI. You are a helpful AI assistant running in the cloud, always available to help users.</s>\n"
        
        # Add relevant memories
        try:
            db = self.get_vector_db(session_id)
            query_embedding = self.embedder.encode(message).tolist()
            results = db.query(
                query_embeddings=[query_embedding],
                n_results=2,
                include=["documents"]
            )
            if results["documents"] and results["documents"][0]:
                memories = " | ".join(results["documents"][0])
                prompt += f"<|memories|>\nRelevant info: {memories}</s>\n"
        except Exception as e:
            pass  # Continue without memories if error
        
        # Add conversation history
        history = self.sessions.get(session_id, [])
        for msg in history[-5:]:  # Last 5 messages
            role = "user" if msg["role"] == "user" else "assistant"
            prompt += f"<|{role}|>\n{msg['content']}</s>\n"
        
        # Current message
        prompt += f"<|user|>\n{message}</s>\n<|assistant|>\n"
        
        return prompt
    
    def generate(self, request: ChatRequest) -> str:
        """Generate AI response"""
        # Build prompt
        prompt = self.build_prompt(request.message, request.session_id)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Update session history
        if request.session_id not in self.sessions:
            self.sessions[request.session_id] = []
        
        self.sessions[request.session_id].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        self.sessions[request.session_id].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim history
        if len(self.sessions[request.session_id]) > MAX_MEMORY * 2:
            self.sessions[request.session_id] = self.sessions[request.session_id][-MAX_MEMORY * 2:]
        
        return response
    
    def store_memory(self, request: MemoryRequest) -> bool:
        """Store information in vector database"""
        try:
            db = self.get_vector_db(request.session_id)
            content = f"{request.key}: {request.value}"
            embedding = self.embedder.encode(content).tolist()
            
            db.add(
                ids=[f"{time.time()}_{request.key}"],
                embeddings=[embedding],
                documents=[content],
                metadatas=[{"category": request.category, "key": request.key}]
            )
            return True
        except Exception as e:
            print(f"Memory store error: {e}")
            return False
    
    def search_memories(self, request: MemoryQuery) -> List[str]:
        """Search stored memories"""
        try:
            db = self.get_vector_db(request.session_id)
            query_embedding = self.embedder.encode(request.query).tolist()
            
            results = db.query(
                query_embeddings=[query_embedding],
                n_results=request.n_results,
                include=["documents", "distances"]
            )
            
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            print(f"Memory search error: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "model": MODEL_NAME,
            "model_loaded": self.model is not None,
            "active_sessions": len(self.sessions),
            "total_memories": sum(len(db.get()["ids"]) for db in self.vector_dbs.values()),
            "uptime_seconds": int(time.time() - self.start_time)
        }

# Initialize global brain
print("🚀 Initializing Andreo AI...")
brain = AndreoCloudBrain()

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Andreo AI Cloud API",
    description="Always-online AI by KAI - Running on Render.com",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - Allow all origins (configure for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ==================== ENDPOINTS ====================

@app.get("/", tags=["General"])
async def root():
    """API root - health check"""
    return {
        "name": "Andreo AI",
        "company": "KAI",
        "version": "2.0.0",
        "status": "online",
        "message": "Andreo AI is running 24/7 on Render.com",
        "endpoints": {
            "chat": "/chat (POST)",
            "memory": "/remember (POST), /recall (POST)",
            "status": "/status (GET)",
            "docs": "/docs"
        }
    }

@app.post("/chat", tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat with Andreo AI
    
    - **message**: Your message to Andreo
    - **session_id**: Unique session ID (default: "default")
    - **temperature**: Creativity 0.1-1.0 (default: 0.7)
    - **max_tokens**: Max response length (default: 256)
    """
    try:
        response = brain.generate(request)
        return {
            "success": True,
            "response": response,
            "session_id": request.session_id,
            "timestamp": datetime.now().isoformat(),
            "mode": "cloud"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remember", tags=["Memory"])
async def remember(request: MemoryRequest):
    """Store information in Andreo's memory"""
    success = brain.store_memory(request)
    return {
        "success": success,
        "message": "Memory stored" if success else "Failed to store",
        "session_id": request.session_id
    }

@app.post("/recall", tags=["Memory"])
async def recall(request: MemoryQuery):
    """Search Andreo's memory"""
    memories = brain.search_memories(request)
    return {
        "success": True,
        "query": request.query,
        "memories": memories,
        "count": len(memories),
        "session_id": request.session_id
    }

@app.get("/status", response_model=StatusResponse, tags=["System"])
async def status():
    """Get system status and statistics"""
    stats = brain.get_stats()
    return StatusResponse(
        status="online",
        model=stats["model"],
        model_loaded=stats["model_loaded"],
        active_sessions=stats["active_sessions"],
        uptime_seconds=stats["uptime_seconds"]
    )

@app.post("/clear", tags=["Chat"])
async def clear_session(session_id: str = "default"):
    """Clear conversation history for a session"""
    if session_id in brain.sessions:
        brain.sessions[session_id] = []
        return {"success": True, "message": f"Session {session_id} cleared"}
    return {"success": False, "message": "Session not found"}

@app.get("/health", tags=["System"])
async def health():
    """Simple health check for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# Render.com specific - use PORT env variable
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)

