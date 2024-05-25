import modal
from modal import Image, build, enter, exit, method, gpu

GPU_CONFIG = gpu.T4()
INDEX_ROOT = "/vol/colbert/indexes"
INDEX_NAME = "book-of-mormon-index"

lds_rag_image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("colbert-ai", "torch")
)

app = modal.App(name="lds-rag")

volume = modal.Volume.from_name("ragatouille")


@app.cls(image=lds_rag_image, volumes={"/vol": volume})
class RAG:
    @build()
    def download_model(self):
        from colbert import Searcher

        Searcher(
            index=INDEX_NAME,
            index_root=INDEX_ROOT,
        )

    @enter()
    def setup(self):
        from colbert import Searcher

        self.searcher = Searcher(
            index=INDEX_NAME,
            index_root=INDEX_ROOT,
        )

    @method()
    def rag(self, query: str, k: int = 5):
        import math

        print(f"Query={query}")
        print(len(self.searcher.collection[1]))
        k = min(int(k), 100)
        pids, ranks, scores = self.searcher.search(query, k=k)
        pids, ranks, scores = pids[:k], ranks[:k], scores[:k]
        print(pids)
        passages = [self.searcher.collection[pid] for pid in pids]
        print(passages)
        probs = [math.exp(score) for score in scores]
        probs = [prob / sum(probs) for prob in probs]
        topk = []
        for pid, rank, score, prob in zip(pids, ranks, scores, probs):
            text = self.searcher.collection[pid]
            d = {"text": text, "pid": pid, "rank": rank, "score": score, "prob": prob}
            topk.append(d)
        topk = list(sorted(topk, key=lambda p: (-1 * p["score"], p["pid"])))
        return {"query": query, "topk": topk}

        # results = self.model.search(query=query, k=k)
        # return [f"{result['document_id']}: {result['content']}" for result in results]


@app.local_entrypoint()
def main():
    model = RAG()
    print(model.rag.remote(query="What is the Spirit?", k=3))
