import json
import os

class Generator():
    def __init__(self, client, logger, args):
        self.args = args
        self.logger = logger
        self.client = client
        self.usr_name = args.usr_name
        self.agent_name = args.agent_name

    def select_prompts(self, inquiry, context, memories):
        """ 
        Constructs the System and User Prompts.
        Adopts the structure from LD-Agent for fair comparison.
        """
        
        # System Prompt
        sys_prompt = f"As a communication expert with outstanding communication habits, you embody the role of {self.agent_name} throughout the following dialogues.\n"

        # User Prompt (Integrating Context + Retrieved Memory)
        user_prompt = f"""<CONTEXT>\nDrawing from your recent conversation with {self.usr_name}:\n{context}\n""" \
                        + f"""<MEMORY>\nThe memories linked to the ongoing conversation are:\n{memories}\n""" \
                        + f"""\nNow, please role-play as {self.agent_name} to continue the dialogue between {self.agent_name} and {self.usr_name}.\n""" \
                        + f"""{self.usr_name} said: {inquiry}\n""" \
                        + f"""Please respond to {self.usr_name}'s statement:\n"""

        return sys_prompt, user_prompt

    def response_build(self, inquiry, context, memories):
        sys_prompt, user_prompt = self.select_prompts(inquiry, context, memories)
        
        # Invoke LLM Client
        response = self.client.employ(sys_prompt, user_prompt, "ResponseGenerator")
        
        full_prompt_log = f"--- SYSTEM PROMPT ---\n{sys_prompt}\n\n--- USER PROMPT ---\n{user_prompt}"
        return response, full_prompt_log