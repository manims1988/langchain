from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model = "llama-3.3-70b-versatile")

prompt = PromptTemplate(
    template = "generate 1 facts about a topic {topic}",
    input_variables = ['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser 
result = chain.invoke({'topic': 'aliens'})

print(result)