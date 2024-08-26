from fastapi import FastAPI

from src.api.controllers import (
    dataset_controller,
    health_controller,
    train_controller
)
from src.handlers.exception_handler import add_exception_handlers

tags_metadata = [
    {
        "name": "health",
        "description": "checks the health of the API services",
    },
    {
        "name": "dataset",
        "description": "Dataset Tool: Process dataset",
    },
    {
        "name": "train",
        "description": "Train StyleGAN3'",
    }
]


app = FastAPI(
    version="1.0",
    title="StyleGAN3 API",
    description="API for using StyleGAN3",
    openapi_tags=tags_metadata,
)

app.include_router(
    health_controller.router,
    prefix="/health",
    tags=["health"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    dataset_controller.router,
    prefix="/dataset",
    tags=["dataset"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    train_controller.router,
    prefix="/train",
    tags=["train"],
    responses={404: {"description": "Not found"}},
)


add_exception_handlers(app=app)

# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)