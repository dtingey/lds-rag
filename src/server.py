import os
import logging
import qdrant_client
from llama_index.llms.ollama import Ollama
from llama_index.core import (
    StorageContext,
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from json_reader import CustomJSONReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.postprocessor.colbert_rerank import ColbertRerank
import litserve as ls

DOCUMENT_PATH = (
    "/Users/damontingey/personal/lds-rag/data/scriptures/flat/book-of-mormon-flat.json"
)


class ScriptureChatAPI(ls.LitAPI):
    def setup(self, device):
        # self.model = lambda x: int(x) ** 2
        # Settings.llm = Ollama(model="llama3.1:latest", request_timeout=120.0)
        # Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
        # client = qdrant_client.QdrantClient(host="localhost", port=6333)
        # vector_store = QdrantVectorStore(
        #     client=client, collection_name="scripture_search_collection"
        # )
        # storage_context = StorageContext.from_defaults(vector_store=vector_store)
        documents = CustomJSONReader().load_data(DOCUMENT_PATH)[:1000]
        # index = VectorStoreIndex.from_documents(
        #     documents, storage_context=storage_context
        # )
        # self.query_engine = index.as_query_engine()
        Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

        client = qdrant_client.QdrantClient(host="localhost", port=6333)

        vector_store = QdrantVectorStore(
            client=client, collection_name="new_collection"
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        self.retriever = index.as_retriever(similarity_top_k=100)

        self.reranker = ColbertRerank(
            top_n=5,
            model="colbert-ir/colbertv2.0",
            tokenizer="colbert-ir/colbertv2.0",
            keep_retrieval_score=True,
        )

    def decode_request(self, request):
        return request["query"]

    def predict(self, query):
        results = self.retriever.retrieve(query)
        rerank = self.reranker.postprocess_nodes(results, query_str=query)
        return [
            f"Score: {item.score}, {item.metadata["reference"]}: {item.text}"
            for item in rerank
        ]

    def encode_response(self, output):
        return {"output": output}


if __name__ == "__main__":
    api = ScriptureChatAPI()
    server = ls.LitServer(api)
    server.run(port=8000)
