from langchain_core.prompts import PromptTemplate

dynamic_prompt = PromptTemplate( template="write a single line about {topic} in a {style} style", input_variables=["topic", "style"])

prompt_text_1 = dynamic_prompt.format( topic="AI", style="humorous")
prompt_text_2 = dynamic_prompt.format( topic="blockchain", style="formal")

print(prompt_text_1)
print(prompt_text_2)