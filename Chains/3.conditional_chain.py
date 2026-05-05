from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field 
from typing import Literal

load_dotenv()

model1 = ChatGroq(model = "llama-3.3-70b-versatile")

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description = 'the sentiment for the feedback, must be positive or negative')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='classity the sentiment of the following text into positive or negative {feedback}, {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model1 | parser2

prompt2 = PromptTemplate(
    template='write an appropriate response to this positive feedback {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='write an appropriate response to this negative feedback {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment=='positive', prompt2 | model1 | parser),
    (lambda x:x.sentiment=='negative', prompt3 | model1 | parser),
    RunnableLambda(lambda x: 'the sentiment is neutral')
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback':'this is a beautifule place which has a lot of amaxing historical insights and the people are also very good. thank you for having us here'})
print(result)
