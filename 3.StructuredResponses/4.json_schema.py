from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile")

json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "write down all the key themes discussed in the review in a list"
        },
        "summary": {
            "type": "string",
            "descripion": "a brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos","neg"],
            "description": "write down all the pros inside a list"
        },
        "cons": {
            "type": ["array", "null"],
            "items": { "type": "string"},
            "description": "write down all the cons inside a list"
        },
        "name": {
            "type": [ "string", "null"],
            "description": "write the name of the reviewer"
        }
    },
    "required": [ "key_themes", "summary", "sentiment" ]
}

structured_model = model.with_structured_output(json_schema, strict = True)
prompt = """
    The Google Pixel 9 is as close to the ideal Android iPhone experience as one can fathom right now. 
    If Apple phones are the gold standard on how to harmonize hardware and software for a satisfying user experience, then the Pixel 9 is now nearly at that level.  
"""
response = structured_model.invoke(prompt)
print(response)