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
		"options": { "num_predict": 200 }
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
            # Return memory usage in bytes
            size = model.get('size', 0)
            size_vram = model.get('size_vram', 0)
            return size, size_vram
    return 0, 0

def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0: 'Bytes', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size >= power and n < 4:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"

def rate_local_llm_output_with_chatgpt(local_model_output, prompt):
    if os.getenv('OPENAI_API_KEY') is None:
	    return 0
    """Ask OpenAI's ChatGPT to rate the local LLM's output."""
    chatgpt_prompt = f"""
You are a code reviewer. 
Please rate the following response from a local LLM model based on how accurately and effectively it answers the given coding-related prompt.
Please be critical.

Prompt: {prompt}

Local LLM's Response: {local_model_output}

Provide a score from 0 to 100 (0 being nonsense and 100 being perfect).
Please respond only with the score.
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Adjust to any model you prefer
            messages=[
                {"role": "user", "content": chatgpt_prompt}
            ]
        )
        rating = response.choices[0].message.content.strip()
        return rating
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
    response = requests.post("http://localhost:11434/api/pull", json=payload, stream=False)
    # Decode response content and check if it contains the success status
    if b'"status":"success"' in response.content:
        print(f"Model {model_name} pulled successfully.")
    else:
        print(f"Failed to pull model {model_name}: {response.content}")
    
    # Check if the response status code is not 200
    if response.status_code != 200:
        print(f"Failed to pull model {model_name}: {response.text}")

def ensure_models_downloaded(models):
    local_models = get_local_models()
    for model in models:
        if model not in local_models:
            print("Pulling model: " + model)
            pull_model(model)

def load_model(model_name):
    payload = {
        "model": model_name,
        "prompt": "",
        "keep_alive": 3600  # Keep model alive for 1 hour
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        if response.status_code != 200:
            print(f"Error loading model {model_name}: {response.status_code}, {response.text}")
            return False
        else:
            #print(f"Model {model_name} loaded successfully.")
            return True
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return False

def unload_model(model_name):
    payload = {
        "model": model_name,
        "prompt": "",
        "keep_alive": 0  # Unload model immediately
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        if response.status_code != 200:
            print(f"Error unloading model {model_name}: {response.status_code}, {response.text}")
            return False
        else:
            #print(f"Model {model_name} unloaded successfully.")
            return True
    except Exception as e:
        print(f"Failed to unload model {model_name}: {e}")
        return False

# New function to save results to a text file
def save_results_to_file(file_name, model_name, memory_usage, tokens_per_second, rating):
    with open(file_name, 'a') as f:  # 'a' mode appends to the file without overwriting
        f.write(f"{model_name} | {memory_usage} | {tokens_per_second:.2f} | {rating:.2f}\n")


def main():
    models = [
    "yi-coder:1.5b-chat-q2_K",
    "yi-coder:1.5b-chat-q3_K_S",
    "yi-coder:1.5b-chat-q3_K_M",
    "yi-coder:1.5b-chat-q3_K_L",
    "yi-coder:1.5b-chat-q4_0",
    "yi-coder:1.5b-chat-q4_1",
    "yi-coder:1.5b-chat-q4_K_S",
    "yi-coder:1.5b-chat-q4_K_M",
    "yi-coder:1.5b-chat-q5_0",
    "yi-coder:1.5b-chat-q5_1",
    "yi-coder:1.5b-chat-q5_K_S",
    "yi-coder:1.5b-chat-q5_K_M",
    "yi-coder:1.5b-chat-q6_K",
    "yi-coder:1.5b-chat-q8_0",
    "yi-coder:1.5b-chat-fp16",
    "yi-coder:9b-instruct-q2_K",
    "yi-coder:9b-instruct-q3_K_S",
    "yi-coder:9b-instruct-q3_K_M",
    "yi-coder:9b-instruct-q3_K_L",
    "yi-coder:9b-instruct-q4_0",
    "yi-coder:9b-instruct-q4_1",
    "yi-coder:9b-instruct-q4_K_S",
    "yi-coder:9b-instruct-q4_K_M",
    "yi-coder:9b-instruct-q5_0",
    "yi-coder:9b-instruct-q5_1",
    "yi-coder:9b-instruct-q5_K_S",
    "yi-coder:9b-instruct-q5_K_M",
    "yi-coder:9b-instruct-q6_K",
    "yi-coder:9b-instruct-q8_0",
    "yi-coder:9b-instruct-fp16",
    "qwen2.5:0.5b-instruct-q2_K",
    "qwen2.5:0.5b-instruct-q3_K_S",
    "qwen2.5:0.5b-instruct-q3_K_M",
    "qwen2.5:0.5b-instruct-q3_K_L",
    "qwen2.5:0.5b-instruct-q4_0",
    "qwen2.5:0.5b-instruct-q4_1",
    "qwen2.5:0.5b-instruct-q4_K_S",
    "qwen2.5:0.5b-instruct-q4_K_M",
    "qwen2.5:0.5b-instruct-q5_0",
    "qwen2.5:0.5b-instruct-q5_1",
    "qwen2.5:0.5b-instruct-q5_K_S",
    "qwen2.5:0.5b-instruct-q5_K_M",
    "qwen2.5:0.5b-instruct-q6_K",
    "qwen2.5:0.5b-instruct-q8_0",
    "qwen2.5:0.5b-instruct-fp16",
    "qwen2.5:1.5b-instruct-q2_K",
    "qwen2.5:1.5b-instruct-q3_K_S",
    "qwen2.5:1.5b-instruct-q3_K_M",
    "qwen2.5:1.5b-instruct-q3_K_L",
    "qwen2.5:1.5b-instruct-q4_0",
    "qwen2.5:1.5b-instruct-q4_1",
    "qwen2.5:1.5b-instruct-q4_K_S",
    "qwen2.5:1.5b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q5_0",
    "qwen2.5:1.5b-instruct-q5_1",
    "qwen2.5:1.5b-instruct-q5_K_S",
    "qwen2.5:1.5b-instruct-q5_K_M",
    "qwen2.5:1.5b-instruct-q6_K",
    "qwen2.5:1.5b-instruct-q8_0",
    "qwen2.5:1.5b-instruct-fp16",
    "qwen2.5:3b-instruct-q2_K",
    "qwen2.5:3b-instruct-q3_K_S",
    "qwen2.5:3b-instruct-q3_K_M",
    "qwen2.5:3b-instruct-q3_K_L",
    "qwen2.5:3b-instruct-q4_0",
    "qwen2.5:3b-instruct-q4_1",
    "qwen2.5:3b-instruct-q4_K_S",
    "qwen2.5:3b-instruct-q4_K_M",
    "qwen2.5:3b-instruct-q5_0",
    "qwen2.5:3b-instruct-q5_1",
    "qwen2.5:3b-instruct-q5_K_S",
    "qwen2.5:3b-instruct-q5_K_M",
    "qwen2.5:3b-instruct-q6_K",
    "qwen2.5:3b-instruct-q8_0",
    "qwen2.5:3b-instruct-fp16",
    "qwen2.5:7b-instruct-q2_K",
    "qwen2.5:7b-instruct-q3_K_S",
    "qwen2.5:7b-instruct-q3_K_M",
    "qwen2.5:7b-instruct-q3_K_L",
    "qwen2.5:7b-instruct-q4_0",
    "qwen2.5:7b-instruct-q4_1",
    "qwen2.5:7b-instruct-q4_K_S",
    "qwen2.5:7b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q5_0",
    "qwen2.5:7b-instruct-q5_1",
    "qwen2.5:7b-instruct-q5_K_S",
    "qwen2.5:7b-instruct-q5_K_M",
    "qwen2.5:7b-instruct-q6_K",
    "qwen2.5:7b-instruct-q8_0",
    "qwen2.5:7b-instruct-fp16"
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
        print(f"\nLoading model: {model_name}")
        if not load_model(model_name):
            print(f"Failed to load model {model_name}. Skipping.")
            continue
        # Get memory usage
        memory_usage, vram_usage = get_model_memory_usage(model_name)
        memory_str = format_bytes(memory_usage)
        print(f"Memory usage for model {model_name}: {format_bytes(memory_usage)}")
        if vram_usage:
            print(f"VRAM usage for model {model_name}: {format_bytes(vram_usage)}")
        tokens_per_second_list = []
        tokens_per_second_list = []
        ratings_list = []
        ratings_list = []

        print(f"Testing model: {model_name}")

        for prompt in prompts:
            tokens_per_second, model_output = send_prompt(model_name, prompt)
            print(f"Prompt: {prompt} \nResponse: {model_output}")
            if tokens_per_second is None:
                # Assume a timeout or error occurred, skip this model
                print(f"Skipping model {model_name} due to timeout or error.")
                break
            else:
                tokens_per_second_list.append(tokens_per_second)
                # Send the output to ChatGPT for rating
                rating_text = rate_local_llm_output_with_chatgpt(model_output, prompt)
                print(f"Prompt: {prompt}")
                print(f"TPS: {tokens_per_second}")
                print(f"RATING: {rating_text}")
                # Try to extract the numeric rating
                try:
                    rating_value = float(rating_text)
                    ratings_list.append(rating_value)
                except ValueError:
                    print(f"Failed to parse rating '{rating_text}' as a number.")
                    ratings_list.append(0)  # Assign 0 if parsing fails

        if tokens_per_second_list and ratings_list:
            avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list)
            avg_rating = sum(ratings_list) / len(ratings_list)
            print(f"Model: {model_name}")
            print(f"Average Tokens Per Second: {avg_tokens_per_second:.2f}")
            print(f"Average GPT Rating: {avg_rating:.2f}")
            print("=" * 50)
            save_results_to_file("model_test_results.txt", model_name, memory_str, avg_tokens_per_second, avg_rating)
        else:
            print(f"Model {model_name} did not complete successfully.")

        # Unload the model
        unload_model(model_name)

if __name__ == "__main__":
    main()
