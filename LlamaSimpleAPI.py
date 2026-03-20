import logging
import requests

# A simple API wrapper for Llama CPP.
# Using the last 5 Messages as context, and the system with message for the first message.

class LlamaSimpleAPI:
    def __init__(self, sysprompt, url, model):
        self.sysprompt = sysprompt
        self.url = url
        self.url_chat = f"{url}/v1/chat/completions"
        self.model = model
        self.messages_archive = []
        self.context = ""
        self.check_api()
        logging.info(f"LlamaSimpleAPI initialized:\n model: {model} \n url: {url}")


    def check_api(self):
        try:
            for candidate in (f"{self.url}/health", f"{self.url}/v1/models"):
                response = requests.get(candidate, timeout=10)
                if response.ok:
                    logging.info("Llama API is reachable via %s", candidate)
                    return True

            logging.error("Llama API reachable, but no known probe endpoint answered successfully")
            return False
        except requests.exceptions.RequestException as e:
            logging.error(f"Error connecting to Llama API: {e}")
            return False
    
    def set_context(self, context):
        if context:
            self.context = context
        else:
            self.context = ""

    def get_payload(self, user_text,temperature=0.1, max_tokens=750):
        messages_archive_payload = self.messages_archive[-5:]
        system_content = self.sysprompt
        if self.context:
            system_content = f"{self.sysprompt}\n\nContext:\n{self.context}"

        messages_to_send = [
            {"role": "system", "content": system_content},
            *messages_archive_payload,
            {"role": "user", "content": user_text},
        ]
        
        return {
            "model": self.model,
            "messages": messages_to_send,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def ask(self, user_text, temperature=0.1, max_tokens=750) -> str:
        self.check_api()   
        payload = self.get_payload(user_text, temperature, max_tokens)
        timeout_time = len(user_text) * 0.1 + 300
        response = requests.post(self.url_chat, json=payload, timeout=timeout_time)
        logging.info("POST request url: %s", self.url_chat.strip())
        if not response.ok:
            logging.error(f"HTTP status: {response.status_code}")
            logging.error(f"Server response: {response.text}")
            response.raise_for_status()

        data = response.json()
        assistant_text = data["choices"][0]["message"]["content"]

        self.messages_archive.append({
            "role": "user",
            "content": user_text,
        })
        self.messages_archive.append({
            "role": "assistant",
            "content": assistant_text,
        })
        return assistant_text
