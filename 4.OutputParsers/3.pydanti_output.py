from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile")

class Person(BaseModel):
    name: str = Field(description = "the person full name")
    age: int = Field(gt = 18, lt = 60, description = "the person age must not be less thatn 18 and greater than 60")
    city: str = Field(description = "the city where the person lives in")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = (
        "give me the name, age and city of a fictional {place} person\n"
        "make sure the age is greater than 18\n"
        "return the response in the following format: \n\n"
        "{format_instruction}\n\n"
    ),
    input_variables = ["place"],
    partial_variables = {
        "format_instruction": parser.get_format_instructions()
    }
)

prompt = template.invoke({"place": "Austrila"})
chain = template | model | parser 
result = chain.invoke({"place": "Austrila"})
print(result)
