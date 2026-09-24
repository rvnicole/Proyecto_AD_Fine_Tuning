from fastapi import FastAPI
from app.routes.llm_routes import router

app = FastAPI()
app.include_router(router)

@app.get("/")
def root():
    return {
        "message": "Hola desde Python"
    }