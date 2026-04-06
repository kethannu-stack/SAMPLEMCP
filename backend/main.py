
# main.py
# This is the FastAPI web server — the entry point of the backend.
# It receives requests from the frontend and triggers the agent pipeline.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import run_study_pipeline
import os

# ─────────────────────────────────────────────
# Create the FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="Multi-Agent Study Research System",
    description="MCP-style multi-agent system for exam preparation",
    version="1.0.0"
)

# ─────────────────────────────────────────────
# CORS Middleware
# Allows the frontend (different origin) to talk to this backend.
# Required for browser security — without this, requests will be blocked.
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request model — what the frontend sends to us
# Pydantic auto-validates this JSON structure.
# ─────────────────────────────────────────────
class StudyRequest(BaseModel):
    topic: str           # e.g., "Photosynthesis"
    mode: str = "deep"   # "quick" or "deep"


# ─────────────────────────────────────────────
# ENDPOINT: Health Check
# Judges can hit /health to confirm server is running.
# ─────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "online", "message": "Multi-Agent Study System is running"}


# ─────────────────────────────────────────────
# ENDPOINT: Generate Study Material
# This is the main endpoint. It:
# 1. Receives a topic + mode
# 2. Runs the full 3-agent pipeline
# 3. Returns structured study material
# ─────────────────────────────────────────────
@app.post("/study")
async def generate_study_material(request: StudyRequest):
    """
    Main endpoint: accepts a topic and study mode,
    runs the multi-agent pipeline, returns structured output.
    """
    # Validate input
    if not request.topic or len(request.topic.strip()) < 2:
        raise HTTPException(status_code=400, detail="Please provide a valid topic.")
    
    if request.mode not in ["quick", "deep"]:
        raise HTTPException(status_code=400, detail="Mode must be 'quick' or 'deep'.")

    topic = request.topic.strip()
    mode = request.mode

    # Run the full agent pipeline (Research → Synthesis → Evaluation)
    result = await run_study_pipeline(topic=topic, mode=mode)
    
    return result


# ─────────────────────────────────────────────
# ENDPOINT: Regenerate weak sections only
# This is the "Regenerate Weak Sections" button feature.
# ─────────────────────────────────────────────
@app.post("/regenerate")
async def regenerate_weak_sections(request: StudyRequest):
    """
    Regenerates only the weak/thin sections identified by the Evaluation Agent.
    Runs a focused deep-dive pipeline on the same topic.
    """
    topic = request.topic.strip()
    # Always use deep mode for regeneration
    result = await run_study_pipeline(topic=topic, mode="deep")
    return result


# ─────────────────────────────────────────────
# Run the server (for local development)
# On Render, the platform handles this automatically.
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
