import json

def get_content_of_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_reasoning_of_data(data, data_label):
    reasoning_list = []
    for record in data:
        raw = record["raw"]
        finetuned = record["finetuned"]
        if reasoning := raw.get("reasoning"):
            raw.update({"data_label": f"{data_label}_raw"})
            reasoning_list.append((reasoning, raw))
        if reasoning := finetuned.get("reasoning"):
            finetuned.update({"data_label": f"{data_label}_finetuned"})
            reasoning_list.append((reasoning, finetuned))
    return reasoning_list
