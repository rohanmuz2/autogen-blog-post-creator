from autogen import AssistantAgent, UserProxyAgent
from fast_depends import use
from config.config import Config
import os

os.environ["AUTOGEN_USE_DOCKER"] = "False"

user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: (x["content"] is not None) and "AIBROSPOD" in x["content"],
)

writer = AssistantAgent(
    name= "writer",
    llm_config = Config().get_llm_config(),
    system_message="""You write Blog post write 
    
                    You have 2 primary task 
                    1) To write a new blog post for a given topic : Use **NEW BLOG POST
                    2) Update an existing blog post using the critic comments : Use **UPDATE BLOG POST

                    **NEW BLOG POST
                    1) Write the article in 300 words.
                    2)Check for moderation and rewrite the article if required.

                    **UPDATE BLOG POST
                    1) Use critic comment to update the article in 500 words
                    2) Chek for moderation and if required rewrite the article
                    3) Add "AIBROSPOD" at the end of the post only after you incorporated the crtic comments.
                  """
)

critic = AssistantAgent(
    name = "critic",
    llm_config = Config().get_llm_config(),
    system_message="""You are a critic agent. You evaluate the quality of a given blog post and provide constructive feedback on how to improve it. Please respond in a clear and concise manner."""
)

def reflection_message(recipient, messages, sender, config):
    print("Reflecting...", "yellow")
    return f"Reflect and provide critique on the following writing. \n\n {messages[-1]['content']}"


user.register_nested_chats(
    [{"recipient": critic, "message": reflection_message, "summary_method": "last_msg", "max_turns": 1}],
    trigger=writer
)

user.initiate_chat(
    writer,
    message="Write a blog post about DeepSeek",
    max_turns=2
)