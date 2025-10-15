from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field, ConfigDict
from typing import List
import torch
import pickle
import logging
from contextlib import asynccontextmanager
import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from model import SymptomRecommender
from langchain_openai import ChatOpenAI

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

llm2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=google_api_key
)

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=openai_api_key
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and artifacts

model = None
symptom_to_idx = None
idx_to_symptom = None
age_bins = None
edge_index = None
device = None
model_config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and artifacts on startup"""
    global model, symptom_to_idx, idx_to_symptom, age_bins, edge_index, device, model_config

    logger.info("Loading model and artifacts...")

    try:
        # Load artifacts
        with open('symptom_to_idx.pkl', 'rb') as f:
            symptom_to_idx = pickle.load(f)

        with open('idx_to_symptom.pkl', 'rb') as f:
            idx_to_symptom = pickle.load(f)

        with open('age_bins.pkl', 'rb') as f:
            age_bins = pickle.load(f)

        with open('model_config.pkl', 'rb') as f:
            model_config = pickle.load(f)

        graph_data = torch.load('graph.pt', weights_only=False)
        edge_index = graph_data.edge_index

        # Initialize model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = SymptomRecommender(
            num_symptoms=model_config['num_symptoms'],
            num_patients=model_config['num_patients'],
            symptom_embed_dim=model_config['symptom_embed_dim']
        ).to(device)

        # Load trained weights
        model.load_state_dict(torch.load('model.pth', map_location=device, weights_only=True))
        model.eval()

        edge_index = edge_index.to(device)

        logger.info(f"Model loaded successfully on {device}")
        logger.info(f"Vocabulary size: {len(symptom_to_idx)}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    # Cleanup (if needed)
    logger.info("Shutting down...")


app = FastAPI(
    title="Symptom Recommendation API",
    description="AI-powered symptom recommendation system using hybrid ML model",
    version="1.0.0",
    lifespan=lifespan
)

class SymptomRecommendation(BaseModel):
    recommendations: List[str] = Field(..., description="List of recommended symptoms")
    query_symptoms: List[str] = Field(..., description="Original query symptoms from user")
    unknown_symptoms: List[str] = Field(default=[], description="Symptoms not found in vocabulary")
    score: float = Field(..., description="Score of the recommendations (0-1)")
    reason: str = Field(..., description="Brief reason for the recommendations (max 200 chars)")

class RecommendRequest(BaseModel):
    """Request model for symptom recommendations"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "male",
                "age": 26,
                "symptoms": ["ท้องแสบ"],
                "top_k": 5
            }
        }
    )

    gender: str = Field(..., description="Patient gender: 'male' or 'female'")
    age: int = Field(..., description="Patient age", ge=0, le=120)
    symptoms: List[str] = Field(..., description="List of initial symptoms (Thai or English)")
    top_k: int = Field(5, description="Number of recommendations to return", ge=1, le=20)


class RecommendResponse(BaseModel):
    """Response model for symptom recommendations"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "recommendations": ["จุกหน้าอก", "หายใจติดขัด", "ปวดข้อ", "คลื่นไส้", "ปวดศีรษะ"],
                "query_symptoms": ["ท้องแสบ"],
                "unknown_symptoms": []
            }
        }
    )

    recommendations: List[str] = Field(..., description="List of recommended symptoms")
    query_symptoms: List[str] = Field(..., description="Input symptoms that were recognized")
    unknown_symptoms: List[str] = Field(..., description="Input symptoms not in vocabulary")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "vocabulary_size": len(symptom_to_idx) if symptom_to_idx else 0
    }


@app.post("/recommend", response_model=RecommendResponse)
async def recommend_symptoms(request: RecommendRequest = Body(...)):
    """
    Recommend next possible symptoms based on patient demographics and initial symptoms

    Args:
        request: RecommendRequest containing gender, age, symptoms, and top_k

    Returns:
        RecommendResponse with recommendations and metadata
    """
    try:
        # Validate gender
        if request.gender.lower() not in ['male', 'female']:
            raise HTTPException(
                status_code=400,
                detail="Gender must be 'male' or 'female'"
            )

        # Validate symptoms
        if not request.symptoms:
            raise HTTPException(
                status_code=400,
                detail="At least one symptom must be provided"
            )

        # Check which symptoms are in vocabulary
        known_symptoms = []
        unknown_symptoms = []

        for symptom in request.symptoms:
            symptom = symptom.strip()
            if symptom in symptom_to_idx:
                known_symptoms.append(symptom)
            else:
                unknown_symptoms.append(symptom)
                logger.warning(f"Unknown symptom: {symptom}")

        # If no known symptoms, return error
        if not known_symptoms:
            raise HTTPException(
                status_code=400,
                detail=f"None of the provided symptoms are in the vocabulary. Unknown: {unknown_symptoms}"
            )

        # Get recommendations
        patient_idx = 0  # Use dummy patient index for inference

        recommendations = model.recommend(
            patient_idx=patient_idx,
            gender=request.gender.lower(),
            age=request.age,
            query_symptoms=known_symptoms,
            symptom_to_idx=symptom_to_idx,
            idx_to_symptom=idx_to_symptom,
            edge_index=edge_index,
            age_bins=age_bins,
            top_k=request.top_k,
            device=device
        )

        logger.info(f"Generated {len(recommendations)} recommendations for {request.gender}/{request.age} with symptoms {known_symptoms}")

        recommendations = RecommendResponse(
            recommendations=recommendations,
            query_symptoms=known_symptoms,
            unknown_symptoms=unknown_symptoms
        )

        structured_llm_default = llm.with_structured_output(SymptomRecommendation)

        # Improved prompt engineering for better LLM evaluation
        prompt = f"""คุณเป็นผู้เชี่ยวชาญด้านการแพทย์ที่ช่วยวิเคราะห์และปรับปรุงคำแนะนำอาการจากระบบ Deep Learning

**ข้อมูลจากผู้ป่วย:**
- เพศ: {request.gender}
- อายุ: {request.age} ปี
- อาการที่ป้อนเข้ามา: {', '.join(known_symptoms) if known_symptoms else 'ไม่มี'}

**คำแนะนำจากระบบ DL:**
{', '.join(recommendations.recommendations) if recommendations.recommendations else 'ไม่มี'}

**ภารกิจของคุณ:**
1. กรองอาการที่ไม่เกี่ยวข้องทางการแพทย์ ซ้ำซ้อน หรือเป็นข้อมูลพื้นฐาน (เช่น "การรักษาก่อนหน้า")
2. แปลคำภาษาอังกฤษเป็นภาษาไทยที่เข้าใจง่าย
3. จัดลำดับตามความสำคัญทางคลินิก
4. เลือก 3-7 อาการที่มีคุณภาพ

**เกณฑ์การให้คะแนน:**
- 0.9-1.0: เกี่ยวข้องสูงมาก
- 0.7-0.89: เกี่ยวข้อง มีบางข้อต้องปรับ
- 0.5-0.69: เกี่ยวข้องบางส่วน
- <0.5: ไม่เกี่ยวข้อง

**รูปแบบการตอบกลับ:**
- recommendations: รายการอาการที่แนะนำ (3-7 อาการ ภาษาไทย)
- query_symptoms: {known_symptoms}
- unknown_symptoms: {unknown_symptoms}
- score: คะแนน 0-1
- reason: เหตุผลสั้นๆ ไม่เกิน 150 ตัวอักษร สรุปใจความสำคัญเท่านั้น

กรุณาวิเคราะห์และให้คำแนะนำที่ดีขึ้น"""

        result = structured_llm_default.invoke(prompt)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/symptoms")
async def list_symptoms(limit: int = 50):
    """
    List available symptoms in vocabulary

    Args:
        limit: Maximum number of symptoms to return

    Returns:
        Dictionary with symptom list and total count
    """
    if symptom_to_idx is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    symptoms = list(symptom_to_idx.keys())[:limit]

    return {
        "symptoms": symptoms,
        "total": len(symptom_to_idx),
        "showing": len(symptoms)
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Symptom Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /recommend": "Get symptom recommendations",
            "GET /symptoms": "List available symptoms",
            "GET /health": "Health check"
        },
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
