from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGroq(model='llama-3.3-70b-versatile')

model2 = ChatGroq(model='llama-3.3-70b-versatile')

prompt1 = PromptTemplate(
    template = "generate short note from following topic {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'generate 2 short questions and answers from the following text {text}',
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = 'merge the provided notes and question answers into single document {notes}, {qa}',
    input_variables = ['notes', 'qa']
)

parser = StrOutputParser()

runnable_chain = RunnableParallel(
    {
        'notes': prompt1 | model1 | parser,
        'qa': prompt1 | model2 | parser
    }
)

merge_chain = prompt3 | model1 | parser

final_chain = runnable_chain | merge_chain

text = """Support Vector Machines (SVM) offer high accuracy and robust generalization by maximizing the margin between classes, making them highly effective for classification and regression tasks. They are especially powerful in high-dimensional spaces, even with limited data samples. 
Key advantages of SVM include
Effective in High Dimensions
Kernel Trick Versatility
Memory Efficiency
Robustness to Overfitting
Versatile Kernels 
Outlier Management"""

result = final_chain.invoke(text)

print(result)