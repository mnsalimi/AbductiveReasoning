from evaluation import evaluate_model
from analyze import analyze_results

model = "Qwen/Qwen2.5-72B-Instruct"
dataset_name = "medqa"
api_key = "hTQSRchoqsaXBEtFp4tG994VgvCVEaoBDuYTPUZTbYdhMFQ4Rc31xYWoHkRfxTAB"

evaluate_model(
    dataset_name=dataset_name,
    model_name=model,
    api_key=api_key,
    max_tokens=15000,
    temperature=0.7,
    use_cache=True
)

analyze_results("medqa", 8)