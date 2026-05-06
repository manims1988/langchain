from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="C:\\Users\\msman\\Downloads\\student_placement_salary_elite_v2.csv")

data = loader.load()
print(data[0])