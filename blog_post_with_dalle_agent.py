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


class BlogPostWithDalleAgent:
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

    def write_blog(self) -> str:

        planner = AssistantAgent(
            name="planner",
            description="This will be use to plan any given task",
            llm_config=Config().get_llm_config(),
            is_termination_msg=lambda x: (x["content"] is not None) and x["content"].rstrip().endswith("AIBROSPOD"),
            system_message="""
                    You are a Planner Agent, specialized in devising structured plans for complex tasks. Given a user query, you must generate a detailed plan following one of these scenarios:

                    1. **NEW BLOG POST:** Create a plan for generating a brand-new blog post.
                    2. **UPDATE BLOG POST:** Create a plan for updating an existing blog post using critic comments.

                    **Available Agents:**

                    - **writer:** Responsible for composing the article.
                    - **image_generator:** Responsible for generating images or cover photos that match the article's theme.

                    **Example:**

                    - **User Query:** "Write me an article on Generative AI."
                    - **Planner Output:**  
                    1. Direct the writer agent to produce an article on Generative AI.  
                    2. Instruct the image_generator agent to create two cover photos inspired by the article's theme.
                    3. Combine both the results and return a json with fields name blogpost and coverImages
                    4. Append the phrase AIBROSPOD only after all planned tasks are complete.

                    Generate a clear, step-by-step plan tailored to the given query.

    """,
        )
        writer = AssistantAgent(
            name="writer",
            llm_config=Config().get_llm_config(),
            system_message="""You write Blog posts""",
            description="This will be use to write a article for a given post",
            is_termination_msg=lambda x: (x["content"] is not None) and x["content"].rstrip().endswith("AIBROSPOD"),
        )

        image_generator = AssistantAgent(
            name="imageGenerator",
            llm_config=Config().get_llm_config(),
            system_message="""Generate 2 cover photos for the given Article using your generate_image tool""",
            human_input_mode="NEVER",
            description="This will be use to create image for a given topic",
            is_termination_msg=lambda x: (x["content"] is not None) and x["content"].rstrip().endswith("AIBROSPOD"),
        )

        user = UserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            is_termination_msg=lambda x: (x["content"] is not None) and x["content"].rstrip().endswith("AIBROSPOD"),
        )

        register_function(
            self.generate_image,
            caller=image_generator,
            executor=user,
            description="This will be use to generate Image for a given topic",
        )
        allowed_transitions = {
            user: [planner],
            planner: [writer, image_generator],
            writer: [planner,image_generator],
            image_generator: [planner, writer],
        }
        groupChat = GroupChat(
            [planner, user, writer, image_generator],
            messages=[],
            max_round=10,
            send_introductions=True,
            allowed_or_disallowed_speaker_transitions=allowed_transitions,
            speaker_transitions_type="allowed",
        )

        manager = GroupChatManager(
            groupchat=groupChat,
            human_input_mode="NEVER",
        )
        user.initiate_chat(
            manager, message=""" Write me an article for Generative AI."""
        )


blog = BlogPostWithDalleAgent()
blog.write_blog()
