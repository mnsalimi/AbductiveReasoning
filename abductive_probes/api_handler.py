import yaml
from typing import Optional
from pydantic import BaseModel

# OpenAI Client
from openai import OpenAI, APIError


def get_model_response(
    judge_model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    retries: int = 3, 
    timeout: int = 40,
    structured_output: bool = False,
    output_schema: Optional[BaseModel] = None,
):
    """
    Sends a POST request to an LLM API using official Python clients.
    Supports structured JSON output for both providers.
    """
    cfg = yaml.safe_load(open("configs/credentials.yaml"))

    provider = "openai" if judge_model.startswith("openai/") else "google"
    provider_cfg = cfg[provider]
    base_url_from_cfg = provider_cfg["base_url"]
    api_key = provider_cfg["api_key"]
    clean_model = judge_model.replace(f"{provider}/", "")

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url_from_cfg,
            max_retries=retries,
            timeout=timeout,
        )

        if structured_output:
            if output_schema is None:
                raise ValueError("output_schema must be provided for structured_output.")
            
            response = client.responses.parse(
                model=clean_model,
                input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                text_format=output_schema,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            content = response.output_parsed.model_dump()
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        else:
            response = client.chat.completions.create(
                model=clean_model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        return content, usage
    
    except APIError as e:
        raise RuntimeError(f"OpenAI API request failed after retries: {e}") from e


if __name__ == "__main__": 
    prompt = "Calculate 5 - 5.4 + 8.8 - 6.4. No explanation."
    out, usage = get_model_response(
        model_name="google/gemini-2.0-flash",
        input_text=prompt,
        max_tokens=250,
        temperature=0,
    )
    print(out)
    print(usage)