import logging
from openai import OpenAI

class OpenAIClient():
    """
    A unified client for interacting with OpenAI models (GPT-4o-mini).
    Replaces the legacy hybrid client.
    """
    def __init__(self, model, logger, args):
        self.args = args
        self.logger = logger
        self.model = args.model if args.model else "gpt-4o-mini"
        
        # API Key handling
        api_key = args.api_key
        if not api_key:
            self.logger.warning("API Key is missing! Please provide it in run_msc.sh or config.")
        
        self.client = OpenAI(api_key=api_key)

    def employ(self, system_prompt, user_prompt, task_name="default"):
        """
        Send request to OpenAI API.
        """
        # self.logger.info(f"[{task_name}] Sending request to {self.model}...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[      
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0, # Deterministic output
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI API Error during {task_name}: {e}")
            return ""