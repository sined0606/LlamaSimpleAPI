

BASE_URL = "http://192.168.2.201:8080"
MODEL = "mistral-7b-instruct-v0.2.Q6_K.gguf"
DEFAULT_INSTRUCTION = "You are a helpful assistant. Answer briefly and clearly."

from LlamaSimpleAPI import LlamaSimpleAPI

# simple chat session with context, to test the context management of the API
# use url for the context file
def context():
    print("Context session started. Type 'exit' or 'quit' to end the session.")
    api = LlamaSimpleAPI(DEFAULT_INSTRUCTION, BASE_URL, MODEL)
    context = []
    context_nio = True
    while context_nio:
        context_nio = False
        context_url= input("Context ' to finish adding context): \n")
        if context_url.lower() in ["exit", "quit"]:
            break
        try:
            with open(context_url, "r") as f:
                context.append(f.read())
        except Exception as e:
            context_nio = True
            print(f"Error reading context file: {e}")
            continue

    api.set_context("\n".join(context))

    while True:
        user_input = input("User: \n")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = api.ask(user_input, temperature=0.0, max_tokens=750)
        print(f"\nAssistant:\n{response}\n")

# simple chat session without context, to test the basic functionality of the API
def chat():
    print("Chat session started. Type 'exit' or 'quit' to end the session.")
    api = LlamaSimpleAPI(DEFAULT_INSTRUCTION, BASE_URL, MODEL)
    while True:
        user_input = input("User: \n")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = api.ask(user_input, temperature=0.0, max_tokens=750)
        print(f"\nAssistant:\n{response}\n")

if __name__ == "__main__":
    # User interface for testing the LlamaSimpleAPI
    print("Welcome to the Llama CPP API Test. Type 'exit' or 'quit' to end the session.")
    print("Type 'char' or 'c' to start a chat session with the assistant.")
    print("Type 'context' or 't' to start a context session with the assistant.")
    messages = []
    while True:
        user_input = input("Input: \n")
        match user_input.lower():
            case "exit" | "quit":
                print("Exiting the chat session. Goodbye!")
                break
            case "char" | "c":
                chat()
            case "context" | "t":
                context()
            case _:
                print("Invalid input. Please type 'exit', 'char', or 'context'.")

