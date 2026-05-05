from langchain_community.document_loaders import TextLoader

loader = TextLoader('C:\\Users\\msman\\Downloads\\learninghub\\langchain\\cricket.txt', encoding = 'utf-8')

docs = loader.load()

print(docs[0].metadata)