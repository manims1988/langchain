from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print("blog post generator")
print("provide ideas or topics for the blog post. Type exit to finish")

topic = input('enter blog post topic: ')
chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("you are a professional blog writer. Help generate informative, engaging and well structured blog post about a {topic}"),
    HumanMessagePromptTemplate.from_template("write a detailed blog post about {topic}")
    ])
chat_history = []

while True:
    user_input = input("ideas or instructions on type exit")
    if user_input.lower() == "exit":
        print("Exiting blog post generator")
        break

    messages = chat_prompt_template.format_messages(topic = topic)

    for previous in chat_history:
        messages.append(previous)

    messages.append(
        HumanMessagePromptTemplate.from_template(user_input).format_messages(user_input=user_input)[0]
    )

    response = chat_model.invoke(messages)
    print('blog post content:\n', response.content)