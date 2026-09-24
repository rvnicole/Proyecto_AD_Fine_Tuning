from fastapi import APIRouter
from app.schemas.prompt_schema import Prompt
from app.schemas.response_model_schema import ResponseModel
from app.services.llm_service import ServiceLLM

router = APIRouter(prefix="/prompt", tags=["Prompt"])

service = ServiceLLM()

@router.post("/", response_model=ResponseModel, status_code=200)
def response_model(data: Prompt):
    return service.generate_response(data)