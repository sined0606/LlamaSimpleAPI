

BASE_URL = "http://192.168.0.51:8080"
MODEL = "mistral-7b-instruct-v0.2.Q6_K.gguf"
DEFAULT_INSTRUCTION = "You are a helpful assistant. Answer briefly and clearly."

from LlamaSimpleAPI import LlamaSimpleAPI

if __name__ == "__main__":
    print("Welcome to the Llama CPP API Test. Type 'exit' or 'quit' to end the session.")
    api = LlamaSimpleAPI(DEFAULT_INSTRUCTION, BASE_URL, MODEL)
    messages = []
    while True:
        user_input = input("User: \n")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = api.ask(user_input, temperature=0.0, max_tokens=750)
        print(f"\nAssistant:\n{response}\n")
