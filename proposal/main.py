from step1 import step1
from loader import load_med_qa_dataset

def main(): 
    model_name = "Llama4-Scout-17B-16E"

    dataset = load_med_qa_dataset(n_samples=2)
    for id, sample in enumerate(dataset):
        if id==0:
            continue
        print("sample")
        print(sample)
        step1_result = step1(sample, id, model_name, "medqa")
        print(step1_result["model_answer"]['nodes'])
        print(step1_result["model_answer"]['edges'])
        print(step1_result["model_answer"]['reasoning'])

if __name__ == "__main__":
    main()
