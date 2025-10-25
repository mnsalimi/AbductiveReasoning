import requests
import json
import time
import yaml
import os

def get_model_response(
    model_name: str,
    api_key: str,
    input_text: str,
    max_tokens: int,
    temperature: float,
    retries: int = None,
    timeout: int = None,
    base_url: str = None
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
        base_url (str): The base URL for the API endpoint.

    Returns:
        str: The content of the model's response.

    Raises:
        RuntimeError: If the request fails after all retries.
    """
    # Load config if parameters not provided
    if retries is None or timeout is None or base_url is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        retries = retries or config["api"]["retries"]
        timeout = timeout or config["api"]["timeout"]
        base_url = base_url or config["api"]["base_url"]
    
    url = base_url

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
            response.raise_for_status()
            response_data = response.json()
            
            # Debug: Print response structure if parsing fails
            try:
                content = response_data['choices'][0]['message']['content']
                usage = response_data['usage']
            except (KeyError, IndexError) as parse_error:
                print(f"\n⚠️  Response parsing failed for model '{model_name}'")
                print(f"Error: {parse_error}")
                print(f"Response keys: {list(response_data.keys())}")
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    print(f"First choice keys: {list(response_data['choices'][0].keys())}")
                    if 'message' in response_data['choices'][0]:
                        print(f"Message keys: {list(response_data['choices'][0]['message'].keys())}")
                else:
                    print(f"Full response structure: {json.dumps(response_data, indent=2)[:1000]}")
                raise
            
            return content, usage
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Attempt {attempt + 1}/{retries} failed: {e}")
            last_exception = e
            if attempt < retries - 1:
                # Exponential backoff with jitter
                sleep_time = (2 ** attempt) + (0.1 * attempt)
                time.sleep(sleep_time)
            # Remove the redundant 'else: continue' as the loop will continue naturally

    raise RuntimeError(f"Failed to get a response after {retries} attempts") from last_exception


if __name__ == "__main__":
    # Load config for test
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    MY_API_KEY = config["api"]["api_key"]
    MY_MODEL = "Qwen/Qwen3-32B"
    MY_PROMPT = "Hi! How are you?"

    model_output, usage = get_model_response(
        model_name=MY_MODEL,
        api_key=MY_API_KEY,
        input_text=MY_PROMPT,
        max_tokens=512,
        temperature=0.8
    )

    print("--- Model Output ---")
    print(model_output)

    print("--- Token Usage ---")
    print(usage)