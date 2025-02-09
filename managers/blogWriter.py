from autogen import (
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
    register_function,
)
from config.config import Config
import os
from openai import OpenAI


client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))
os.environ["AUTOGEN_USE_DOCKER"] = "False"


class BLogWriter:
    def __init__(self) -> None:
        pass

    def generate_image(self, prompt: str):
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url

    def blog_writer_agent(self) -> GroupChatManager:

        blog_assistant = AssistantAgent(
            name="blogAssistant",
            llm_config=Config().get_llm_config(),
            system_message="""You are a Blog Assistant agent. Your task is to generate an engaging blog post and a matching cover image based on a provided article theme. Use your generate_image tool to create the cover photo. Once finished, return both the blog post text and the image URL in a JSON object with the keys 'blog_post', 'image_url' and 'stop_code'= AIBROSPOD """,
            description="This agent writes blog posts and generates cover images for a given article theme.",
            is_termination_msg=lambda x: (x["content"] is not None) and "AIBROSPOD" in x["content"],
        )

        image_generator_executor = AssistantAgent(
            name="imageGenerator",
            llm_config=Config().get_llm_config(),
            system_message="""You design beutifull images""",
            human_input_mode="NEVER",
            description="This will be use to create image for a given topic",
            is_termination_msg=lambda x: (x["content"] is not None) and "AIBROSPOD" in x["content"]
            and x["content"].rstrip().endswith("AIBROSPOD"),
        )

        register_function(
            self.generate_image,
            caller=blog_assistant,
            executor=image_generator_executor,
            description="This will be use to generate Image for a given topic",
        )
        groupChat = GroupChat(
            [blog_assistant, image_generator_executor],
            messages=[],
            max_round=4,
            send_introductions=True,
            speaker_selection_method="round_robin",
        )

        manager = GroupChatManager(
            groupchat=groupChat,
            human_input_mode="NEVER"
        )
        return manager
