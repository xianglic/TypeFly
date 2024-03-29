import os, json
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.organization = os.environ.get('OPENAI_ORGANIZATION')
openai.api_key = os.environ.get('OPENAI_API_KEY')
# MODEL_NAME = "gpt-3.5-turbo-16k"
MODEL_NAME = "gpt-4"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
chat_log_path = os.path.join(CURRENT_DIR, "assets/chat_log.txt")

class LLMWrapper:
    def __init__(self, temperature=0.0):
        self.temperature = temperature
        # clean chat_log
        open(chat_log_path, "w").close()

    def request(self, prompt, model_name=MODEL_NAME):
        response = openai.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        # save the message in a txt
        with open(chat_log_path, "a") as f:
            f.write(prompt + "\n---\n")
            f.write(response.model_dump_json(indent=2) + "\n---\n")

        # print(f"LLM response: {response}")
        return response.choices[0].message.content