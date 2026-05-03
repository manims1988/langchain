from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile")

class Schema(TypedDict):
    key_themes: Annotated[list[str], "must write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "must write down a bried summary of the review"]
    sentiment: Annotated[str, "must return sentiment of the review, either positive or negative"]
    pros: Annotated[Optional[list[str]], "write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "write down all the cons inside a list"]

structured_model = model.with_structured_output(Schema)
prompt = """
    The Google Pixel 9 is as close to the ideal Android iPhone experience as one can fathom right now. 
    If Apple phones are the gold standard on how to harmonize hardware and software for a satisfying user experience, then the Pixel 9 is now nearly at that level.
"""

response = structured_model.invoke(prompt)
print(response)
