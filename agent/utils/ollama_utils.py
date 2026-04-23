import json
import logging
import os

import requests

logger = logging.getLogger(__name__)


def get_ollama_base_url():
    url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    if url.endswith("/v1"):
        url = url[:-3]
    return url.rstrip("/")


def is_ollama_running():
    base_url = get_ollama_base_url()
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def is_model_available(model_name):
    if model_name.startswith("ollama/"):
        model_name = model_name.replace("ollama/", "", 1)

    base_url = get_ollama_base_url()
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code != 200:
            return False

        models = response.json().get("models", [])
        available_names = []
        for m in models:
            name = m.get("name")
            available_names.append(name)
            if ":" in name:
                available_names.append(name.split(":")[0])

        return model_name in available_names
    except (requests.exceptions.RequestException, ValueError, KeyError):
        return False


async def pull_ollama_model(model_name, prompt_session):
    raw_model_name = model_name
    if raw_model_name.startswith("ollama/"):
        raw_model_name = raw_model_name.replace("ollama/", "", 1)

    base_url = get_ollama_base_url()
    print(f"\nPulling Ollama model '{raw_model_name}'...")

    try:
        response = requests.post(
            f"{base_url}/api/pull",
            json={"name": raw_model_name},
            stream=True,
            timeout=None,
        )

        if response.status_code != 200:
            print(f"Error: Failed to pull model (status {response.status_code})")
            return False

        last_status = None
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get("status")

                if status and status != last_status:
                    if "downloading" in status.lower() and "total" in data:
                        completed = data.get("completed", 0)
                        total = data.get("total", 1)
                        percent = (completed / total) * 100
                        print(f"\r  {status}: {percent:.1f}%", end="", flush=True)
                    else:
                        if last_status and "downloading" in last_status.lower():
                            print()
                        print(f"  {status}")
                    last_status = status

        print(f"\nModel '{raw_model_name}' pulled successfully.")
        return True
    except Exception as e:
        print(f"\nError pulling model: {e}")
        return False


async def ensure_ollama_readiness(model_name, prompt_session):
    if not model_name.startswith("ollama/"):
        return True

    if not is_ollama_running():
        print(f"\nError: Ollama server is not reachable at {get_ollama_base_url()}")
        print("Please ensure 'ollama serve' is running.")
        return False

    if not is_model_available(model_name):
        print(f"\nModel '{model_name}' not found locally on Ollama.")
        try:
            from prompt_toolkit.formatted_text import HTML

            answer = await prompt_session.prompt_async(
                HTML(f"Would you like to pull <b>{model_name}</b>? (y/n): ")
            )
            if answer.lower() in ["y", "yes"]:
                return await pull_ollama_model(model_name, prompt_session)
            else:
                return False
        except (EOFError, KeyboardInterrupt):
            return False

    return True
