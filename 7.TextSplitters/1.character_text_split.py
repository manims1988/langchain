from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("C:\\Users\\msman\\Downloads\\LP598201347.pdf")
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    separator = ""
)

# text = 'we are going to learn langchain a lot in this tutorial'
# result = splitter.split_text(text)
result = splitter.split_documents(docs)

#print(result[1].page_content)
print(result[1].metadata)