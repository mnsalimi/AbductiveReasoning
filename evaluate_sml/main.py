from evaluation import evaluate_model
from analyze import analyze_results

model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
dataset_name = "medqa"
api_key = "hTQSRchoqsaXBEtFp4tG994VgvCVEaoBDuYTPUZTbYdhMFQ4Rc31xYWoHkRfxTAB"

evaluate_model(
    dataset_name=dataset_name,
    model_name=model,
    api_key=api_key,
    max_tokens=15000,
    temperature=0.7,
    use_cache=False
)

analyze_results("medqa", 2)