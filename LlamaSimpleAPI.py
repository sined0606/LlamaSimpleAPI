#import logging
import src.app.services.log_md as logging
import os
from pathlib import Path
import requests
import sentencepiece as spm




# A simple API wrapper for Llama CPP.
# Using the last 5 Messages as context, and the system with message for the first message.

class LlamaSimpleAPI:
    
    def __init__(self, sysprompt, url, model,sys_max_tokens=2048):
        self.sysprompt = sysprompt
        self.url = url
        self.url_chat = f"{url}/v1/chat/completions"
        self.model = model
        self.sys_max_tokens = sys_max_tokens
        self.messages_archive = []
        self.context = ""
        self.models_info = None
        self.models = []
        self.models_data = []
        self.selected_model_info = {}
        self.selected_model_data = {}
        self.model_name = ""
        self.model_id = ""
        self.model_object = ""
        self.model_created = None
        self.model_owned_by = ""
        self.model_type = ""
        self.model_description = ""
        self.model_tags = []
        self.model_capabilities = []
        self.model_parameters = ""
        self.model_modified_at = ""
        self.model_size = ""
        self.model_digest = ""
        self.model_parent = ""
        self.model_format = ""
        self.model_family = ""
        self.model_families = []
        self.model_parameter_size = ""
        self.model_quantization_level = ""
        self.model_meta = {}
        self.model_vocab_type = None
        self.model_vocab_size = None
        self.model_context_train = None
        self.model_embedding_size = None
        self.model_parameter_count = None
        self.model_file_size = None
        self.check_api()
        self.get_model_info()

        logging.debug(f"LlamaSimpleAPI initialized:\n model: {model} \n url: {url}")
    
    def get_model_info(self):
        try:
            response = requests.get(f"{self.url}/v1/models", timeout=10)
            if response.ok:
                models_info = response.json()
                self.models_info = models_info
                self.models = models_info.get("models", [])
                self.models_data = models_info.get("data", [])

                selected_model_info = next(
                    (item for item in self.models if item.get("model") == self.model or item.get("name") == self.model),
                    {},
                )
                selected_model_data = next(
                    (item for item in self.models_data if item.get("id") == self.model),
                    {},
                )

                details = selected_model_info.get("details", {})
                meta = selected_model_data.get("meta", {})

                self.selected_model_info = selected_model_info
                self.selected_model_data = selected_model_data
                self.model_name = selected_model_info.get("name", "")
                self.model_id = selected_model_data.get("id", self.model)
                self.model_object = selected_model_data.get("object", "")
                self.model_created = selected_model_data.get("created")
                self.model_owned_by = selected_model_data.get("owned_by", "")
                self.model_type = selected_model_info.get("type", "")
                self.model_description = selected_model_info.get("description", "")
                self.model_tags = selected_model_info.get("tags", [])
                self.model_capabilities = selected_model_info.get("capabilities", [])
                self.model_parameters = selected_model_info.get("parameters", "")
                self.model_modified_at = selected_model_info.get("modified_at", "")
                self.model_size = selected_model_info.get("size", "")
                self.model_digest = selected_model_info.get("digest", "")
                self.model_parent = details.get("parent_model", "")
                self.model_format = details.get("format", "")
                self.model_family = details.get("family", "")
                self.model_families = details.get("families", [])
                self.model_parameter_size = details.get("parameter_size", "")
                self.model_quantization_level = details.get("quantization_level", "")
                self.model_meta = meta
                self.model_vocab_type = meta.get("vocab_type")
                self.model_vocab_size = meta.get("n_vocab")
                self.model_context_train = meta.get("n_ctx_train")
                self.model_embedding_size = meta.get("n_embd")
                self.model_parameter_count = meta.get("n_params")
                self.model_file_size = meta.get("size")

                logging.debug("Available models: %s", models_info)
                return models_info
            else:
                logging.error(f"Failed to get model info. HTTP status: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Error connecting to Llama API: {e}")
            return None
    
    def count_tokens(self, payload):
        text = " ".join([msg["content"] for msg in payload["messages"]])
        tokenizer = spm.SentencePieceProcessor()
        if tokenizer is not None:
            return len(tokenizer.encode(text))

        # Rough fallback to avoid crashing when the tokenizer model is unavailable.
        return max(1, len(text) // 4)

    def calculate_timeout(self, payload):
        prompt_text = " ".join(msg["content"] for msg in payload["messages"])
        prompt_size_bytes = len(prompt_text.encode("utf-8"))
        return prompt_size_bytes * 0.1 + 300

    def check_api(self):
        try:
            for candidate in (f"{self.url}/health", f"{self.url}/v1/models"):
                response = requests.get(candidate, timeout=10)
                if response.ok:
                    logging.debug("Llama API is reachable via %s", candidate)
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

    def get_payload_with_archive(self, user_text,temperature=0.1, max_tokens=750):
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
            "max_tokens": int(max_tokens),
        }

    def get_payload(self, user_text,temperature=0.1, max_tokens=750):
        system_content = self.sysprompt
        if self.context:
            system_content = f"{self.sysprompt}\n\nContext:\n{self.context}"

        messages_to_send = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_text},
        ]
        
        return {
            "model": self.model,
            "messages": messages_to_send,
            "temperature": temperature,
            "max_tokens": int(max_tokens),
        }
    
    def chunking_payload(self, payload, max_tokens):
        chunk_overhead = 500
        prompt_chunking = "The following context is chunked into parts. please make a summery of the content for future use. " \
                        "If the content is not relevant to the question, please ignore it. the summery choud cut the irrelevant part and keep the relevant part. " \
                        "the summery should be concise and only include the relevant information. the summery should be in the format of a list of bullet points. " \
                        "the summry should be in english"

        # Create a single string of all messages content with role tags to preserve the structure
        all_content = " ".join([f'{msg["role"]}: {msg["content"]}' for msg in payload["messages"]])
        n_chunks = (self.count_tokens(payload) // (self.sys_max_tokens - chunk_overhead) + 1)
        number_of_chars = len(all_content)
        chars_per_chunk = number_of_chars // n_chunks
        logging.info(f"Chunking payload into {n_chunks} chunks, each with approximately {chars_per_chunk} characters.")
        chunked_contents = []
        for i in range(n_chunks):
            logging.info(f"Processing chunk {i+1}/{n_chunks}")
            chunked_content = all_content[i*chars_per_chunk:(i+1)*chars_per_chunk]
            answered_chunk = self.ask_single(f"{prompt_chunking}: Context: {chunked_content}", temperature=0.1, max_tokens=chunk_overhead-100)
            chunked_contents.append(answered_chunk)

        self.messages_archive = []
        new_context = "\n".join(chunked_contents)
        self.set_context(new_context)

    def ask(self, user_text, temperature=0.1, max_tokens=750) -> str:
        self.check_api()   
        payload = self.get_payload_with_archive(user_text, temperature, max_tokens)
        prompt_token_count = self.count_tokens(payload)
        available_prompt_tokens = max(1, self.sys_max_tokens - int(max_tokens))
        if prompt_token_count > available_prompt_tokens:
            logging.debug(
                "Payload token count %s exceeds available prompt budget %s, truncating context.",
                prompt_token_count,
                available_prompt_tokens,
            )
            self.chunking_payload(payload, max_tokens)
            payload = self.get_payload_with_archive(user_text, temperature, max_tokens)

        timeout_time = self.calculate_timeout(payload)
        response = requests.post(self.url_chat, json=payload, timeout=timeout_time)
        logging.debug("POST request url: %s", self.url_chat.strip())
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

    def ask_single(self, user_text, temperature=0.1, max_tokens=750) -> str:
        self.check_api()   
        payload = self.get_payload(user_text, temperature, max_tokens)
        timeout_time = self.calculate_timeout(payload)
        response = requests.post(self.url_chat, json=payload, timeout=timeout_time)
        logging.debug("POST request url: %s", self.url_chat.strip())
        if not response.ok:
            logging.error(f"HTTP status: {response.status_code}")
            logging.error(f"Server response: {response.text}")
            response.raise_for_status()

        data = response.json()
        assistant_text = data["choices"][0]["message"]["content"]
        return assistant_text
