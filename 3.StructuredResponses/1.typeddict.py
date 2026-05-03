from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile")

class Review(TypedDict):
    summary: str 
    sentiment: str 

prompt = """
    the hardware is great, but the software feels bloated.
    there are too many pre-installed apps that i never use and can't uninstall
    the battery life is descent, but it could be better
    also the ui looks outdated compared to other brands
    hoping for a software update to fix this. overall, its an average phone with some good features but also some drawbacks
"""

#response = model.invoke(prompt)
structured_model = model.with_structured_output(Review)
response = structured_model.invoke(prompt)
print(response)