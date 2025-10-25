import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI



class LLMInterface:
    
    def __init__(self, model_name: Optional[str] =  None):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def send_message(self, message: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", 
                 "content": f"""{prompt}
                    test : {message} """
                },
            ]
        )

        #only returns the string content of the response in the future implement more complex response handling
        return response.choices[0].message.content
    

#unit test
if __name__ == "__main__":
    llm = LLMInterface(model_name="gpt-3.5-turbo")
    prompt = "Please summarize the following text."
    message = "Artificial Intelligence is transforming the world."
    response = llm.send_message(message, prompt)
    print("LLM Response:", response)