import urllib.request
import json

def test_ollama():
    req = urllib.request.Request("http://localhost:11434/api/tags")
    try:
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            print(f"Ollama models: {models}")
    except Exception as e:
        print(f"Ollama error: {e}")

if __name__ == "__main__":
    test_ollama()
