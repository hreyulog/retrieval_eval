import typer
import time
from evaluation.core.utils.logger import Logger

from evaluation.core.retrieval import RetrievalEvaluator

app = typer.Typer(
    help="RAG Evaluation CLI: Evaluate retrieval and end-to-end performance, including timing"
)


@app.command()
def eval_retrieval(
    dataset: str = typer.Argument(help="Dataset for retrieval evaluation"),
    embedding_model: str = typer.Argument(help="Embedding model path"),
    top_k: int = typer.Option(5, help="Top-K documents to consider for recall"),
):
    """
    Evaluate the retrieval performance of the RAG system.
    Metrics include Recall@K, Precision@K, MRR, etc.
    Also records retrieval time.
    """
    logger = Logger.create(source=dataset)
    logger.info(f"Starting retrieval evaluation on {dataset} with top-{top_k}")

    evaluator = RetrievalEvaluator(dataset=dataset, embedding_model=embedding_model, top_k=top_k)
    metrics_res = evaluator.evaluate()
    
    logger.info(f"Retrieval evaluation completed in {metrics_res['retrieval_time_sec']:.4f}s")

    for metric, value in metrics_res.items():
        logger.info(f"{metric}: {value}")
    filename = f"retrieval_metrics_top{top_k}.json"
    filepath = logger.save_json(metrics_res, filename)
    logger.info(f"Saved retrieval metrics to {filepath}")

@app.command()
def eval_end2end(
    dataset: str = typer.Argument(help="Dataset for end-to-end evaluation"),
    metrics: str = typer.Option("rouge,bleu,meteor", help="Comma-separated list of generation metrics"),
):
    """
    Evaluate the end-to-end performance of the RAG system (retrieval + generation).
    Metrics include ROUGE, BLEU, METEOR, etc.
    Also records end-to-end time.
    """
    logger = Logger.create(source=dataset)
    metric_list = [m.strip() for m in metrics.split(",")]
    logger.info(f"Starting end-to-end evaluation on {dataset} with metrics: {metric_list}")

    start_time = time.time()
    evaluator = EndToEndEvaluator(dataset=dataset, metrics=metric_list)
    results = evaluator.evaluate()
    end_time = time.time()
    
    results["end2end_time_sec"] = end_time - start_time
    logger.info(f"End-to-end evaluation completed in {results['end2end_time_sec']:.4f}s")

    for metric, value in results.items():
        logger.info(f"{metric}: {value}")


if __name__ == "__main__":
    app()
