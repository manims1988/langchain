from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field 

load_dotenv()
model = ChatGroq(model = "llama-3.3-70b-versatile")

class Schema(BaseModel):
    key_themes: list[str] = Field(description="write down the key themes discussed in the review in a list")
    summary: str = Field(description="A bried summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review")
    pros: Optional[list[str]] = Field(default = None, description = "write down all the pros in a list")
    cons: Optional[list[str]] = Field(default = None, description = "write down all the pros in a list")
    name: Optional[str] = Field(default = None, description = "write down the name of the reviewer")

structured_model = model.with_structured_output(Schema, strict = True)

prompt = """ 
    The Google Pixel 9 is as close to the ideal Android iPhone experience as one can fathom right now. 
    If Apple phones are the gold standard on how to harmonize hardware and software for a satisfying user experience, then the Pixel 9 is now nearly at that level.    
"""

response = structured_model.invoke(prompt)
print(response)