import argparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ragatouille import RAGPretrainedModel

import uvicorn

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

path_to_index = ".ragatouille/colbert/indexes/lds-scriptures-index/"
RAG = RAGPretrainedModel.from_index(path_to_index)

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/scripture_references")
def get_scripture_references(question: str, k: int = 5):
    results = RAG.search(query=question, k=5)
    return [f"{result['document_id']}: {result['content']}" for result in results]


if __name__ == "__main__":
    # get port as arg
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5656, help="Port to run server on")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload", default=False
    )
    parser.add_argument("--analysis_model", type=str, help="Analysis model to use: 'gemma' or 'llama3'")
    args = parser.parse_args()

    uvicorn.run("app:app", reload=args.reload, port=args.port)

    
