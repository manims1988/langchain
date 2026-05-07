from langchain_text_splitters import RecursiveCharacterTextSplitter, PythonCodeTextSplitter

text = """
class Student:
    def __init__(self,name,age,grade):
        self.name = name
        self.age = age
        self.grade = grade

    def get_details(self):
        return self.name

    def is_pass(self):
        return self.grade >=6.0

student1 = Student('joe', 20, 8.2)
print(student1.get_detaials())
if student1.is_pass():
    print('the student is paassed')
else:
    print('the student is not passed')
"""

splitter = PythonCodeTextSplitter(
    chunk_size = 300,
    chunk_overlap = 100
)

chunks = splitter.split_text(text)
print(chunks)

