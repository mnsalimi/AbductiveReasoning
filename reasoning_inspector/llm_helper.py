from openai import OpenAI
import httpx
from pydantic import BaseModel
from pydantic import BaseModel, Field
import json
import os

http_client = httpx.Client(timeout=60*2)
api_key = os.getenv("API_KEY")
base_url = "https://gw.ai-platform.ir/v1"
# to list models
# !curl https://gw.ai-platform.ir/models -H "Authorization: Bearer api_key"
client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client, max_retries=1)
print(http_client.timeout)


def ask_llm(system_prompt, user_prompt, model, response_format_model):
    product_schema = response_format_model.model_json_schema()

    product_schema.setdefault("additionalProperties", False)

    response_format = {
        "type": "json_schema",
        "json_schema": {"name": response_format_model.__name__, "strict": True, "schema": product_schema},
    }

    response = client.chat.completions.create(
        model=model,
        response_format=response_format,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    try:
        content = response.choices[0].message.content
        data = json.loads(content)
        return response_format_model(**data)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error parsing JSON: {e}")
        return response_format_model()
