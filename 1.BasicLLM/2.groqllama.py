from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model = "llama-3.1-8b-instant")

response = model.invoke("hello, langchain in one sentence")
print(response)