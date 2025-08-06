import requests
import json
import time

def get_model_response(
    model_name: str,
    api_key: str,
    input_text: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    retries: int = 3,
    timeout: int = 20
):
    """
    Sends a request to the specified model API and retrieves the response.

    Args:
        model_name (str): The name of the model to use.
        api_key (str): The API key for authorization.
        input_text (str): The input prompt for the model.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The sampling temperature.
        retries (int): The number of times to retry the request in case of failure.
        timeout (int): The timeout for the request in seconds.

    Returns:
        str: The content of the model's response.

    Raises:
        RuntimeError: If the request fails after all retries.
    """
    url = "https://gw.ai-platform.ir/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": input_text}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    last_exception = None

    for attempt in range(retries):
        try:
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=timeout
            )
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            response_data = response.json()
            content = response_data['choices'][0]['message']['content']
            return content
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Attempt {attempt + 1}/{retries} failed: {e}")
            last_exception = e
            if attempt < retries - 1:
                # Exponential backoff for retries
                time.sleep(2 ** attempt)
            else:
                # No need to sleep on the last attempt
                continue

    raise RuntimeError(f"Failed to get a response after {retries} attempts") from last_exception


if __name__ == "__main__":
    MY_API_KEY = "hTQSRchoqsaXBEtFp4tG994VgvCVEaoBDuYTPUZTbYdhMFQ4Rc31xYWoHkRfxTAB"
    MY_MODEL = "Qwen/Qwen3-32B"
    MY_PROMPT = "Hi! How are you?"

    model_output = get_model_response(
        model_name=MY_MODEL,
        api_key=MY_API_KEY,
        input_text=MY_PROMPT,
        max_tokens=256,
        temperature=0.8
    )

    print("--- Model Output ---")
    print(model_output)