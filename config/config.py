from math import dist
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self) -> None:
        pass
    def get_llm_config(self) -> str:
        return {
            "config_list": [
                {
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    "api_key": os.environ.get("OPENAI_KEY"),
                }
            ]
        }
    
    

