from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('C:\\Users\\msman\\Downloads\\LP598201347.pdf')
docs = loader.load()

print(docs)