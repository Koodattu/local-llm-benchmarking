import requests
import time
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')  # Read API key from .env file

API_URL = "http://localhost:11434/api/generate"

def calculate_tokens_per_second(eval_count, eval_duration):
    # Calculate tokens per second: tokens / (eval_duration in seconds)
    return eval_count / (eval_duration / 1e9)

def send_prompt(model_name, prompt):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0  # Unload the model immediately after the response
    }
    try:
        # Set timeout to 120 seconds (2 minutes)
        response = requests.post(API_URL, json=payload, timeout=120)
        response_json = response.json()

        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            return None, None

        eval_count = response_json.get("eval_count", 0)
        eval_duration = response_json.get("eval_duration", 0)
        model_output = response_json.get("response", "")

        if eval_count and eval_duration:
            tokens_per_second = calculate_tokens_per_second(eval_count, eval_duration)
            return tokens_per_second, model_output
        else:
            print(f"Error: Missing evaluation data in response for model {model_name}.")
            return None, None

    except requests.exceptions.Timeout:
        print(f"Timeout: Model {model_name} took too long to respond.")
        return None, None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None, None

def get_running_models():
    response = requests.get("http://localhost:11434/api/ps")
    if response.status_code == 200:
        return response.json().get('models', [])
    else:
        print("Failed to retrieve running models.")
        return []

def get_model_memory_usage(model_name):
    running_models = get_running_models()
    for model in running_models:
        if model['name'] == model_name:
            # Return memory usage in bytes (either from 'size' or 'size_vram')
            return model.get('size', 0) or model.get('size_vram', 0)
    return 0

def rate_local_llm_output_with_chatgpt(local_model_output, prompt):
    """Ask OpenAI's ChatGPT to rate the local LLM's output."""
    chatgpt_prompt = f"""
You are a code reviewer. Please rate the following response from a local LLM model based on how accurately and effectively it answers the given coding-related prompt.

Prompt: {prompt}

Local LLM's Response: {local_model_output}

Provide a score from 0 to 100 (0 being nonsense and 100 being perfect).
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Adjust to any model you prefer
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": chatgpt_prompt}
            ]
        )
        rating = response['choices'][0]['message']['content']
        return rating.strip()
    except Exception as e:
        print(f"Failed to get a rating from ChatGPT: {e}")
        return "No rating"

def get_local_models():
    response = requests.get("http://localhost:11434/api/tags")
    if response.status_code == 200:
        models_info = response.json().get('models', [])
        local_models = [model['name'] for model in models_info]
        return local_models
    else:
        print("Failed to retrieve local models.")
        return []

def pull_model(model_name):
    payload = {
        "name": model_name
    }
    response = requests.post("http://localhost:11434/api/pull", json=payload, stream=True)
    if response.status_code == 200:
        print(f"Successfully started pulling model {model_name}")
        # Process streaming response
        for line in response.iter_lines():
            if line:
                status_update = line.decode('utf-8')
                print(status_update)
    else:
        print(f"Failed to pull model {model_name}: {response.text}")

def ensure_models_downloaded(models):
    local_models = get_local_models()
    for model in models:
        if model not in local_models:
            print(f"Model {model} not found locally. Pulling...")
            pull_model(model)
        else:
            print(f"Model {model} is already downloaded.")

def main():
    models = [
        "yi-coder:1.5b-chat-q3_K_S",
        "yi-coder:1.5b-chat-q3_K_M",
        "yi-coder:1.5b-chat-q3_K_L",
        "yi-coder:1.5b-chat-q4_0",
        "yi-coder:1.5b-chat-q4_K_M",
        "yi-coder:1.5b-chat-q8_0"
    ]  # Add your models here

    # Ensure models are downloaded
    ensure_models_downloaded(models)

    # Programming-related prompts for testing code generation and answering questions
    prompts = [
        "Write a Python function to calculate the factorial of a number.",
        "Explain the difference between a list and a tuple in Python.",
        "Write a SQL query to find the second highest salary from an employee table.",
        "How do you implement a binary search algorithm in Python?",
        "What is the time complexity of quicksort, and how does it work?",
    ]

    for model_name in models:
        print(f"Running model: {model_name}")
        tokens_per_second_list = []

        for prompt in prompts:
            print(f"Prompt: {prompt}")
            tokens_per_second, model_output = send_prompt(model_name, prompt)
            if tokens_per_second is None:
                # Assume a timeout or error occurred, skip this model
                print(f"Skipping model {model_name} due to timeout or error.")
                break
            else:
                tokens_per_second_list.append(tokens_per_second)
                # Send the output to ChatGPT for rating
                rating = rate_local_llm_output_with_chatgpt(model_output, prompt)
                print(f"Local LLM's Response: {model_output}")
                print(f"ChatGPT Rating: {rating}")
                print("-" * 50)

        if tokens_per_second_list:
            avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list)
            memory_usage = get_model_memory_usage(model_name)
            memory_in_mb = memory_usage / (1024 * 1024)  # Convert bytes to MB
            print(f"Model: {model_name}, Average Tokens Per Second: {avg_tokens_per_second:.2f}, Memory Usage: {memory_in_mb:.2f} MB")
            print("=" * 50)

if __name__ == "__main__":
    main()
