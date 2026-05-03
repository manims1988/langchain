from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile")

template_1 = PromptTemplate(
    template = "write a detailed report on {topic}",
    input_variables = ["topic"]
)

template_2 = PromptTemplate(
    template = "write a 5 line summary on the following {text}",
    input_variables = ["text"]
)

parser = StrOutputParser()

chain = template_1 | model | parser | template_2 | model | parser 

response = chain.invoke({"topic": "Black Hole"})
print(response)