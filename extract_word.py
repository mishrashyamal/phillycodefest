import PyPDF2
import spacy

pdf_file = open("RISHABH's_RESUME.docx.pdf", 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)

for page_num in range(len(pdf_reader.pages)):
    
    page = pdf_reader.pages[page_num]
    text = page.extract_text()
    text = text.lower()

import spacy

# load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')
import re

# example resume text

# create a Spacy Doc object
doc = nlp(text)

# # Find the start and end indices of the work experience section
# work_exp_start = 0
# work_exp_end = len(text)
# for i, token in enumerate(doc):
#     if token.text.lower() == "work experience" or token.text.lower() == "work" or token.text.lower() == "experience" or token.text.lower() == "professional experience" or token.text.lower() == "relevant experience":
#         work_exp_start = token.idx
#     elif token.text.lower() == "education" or token.text.lower() == "educational background" or token.text.lower() == "education background":
#         work_exp_end = token.idx
#         break

# # Extract the contents of the work experience section
# work_exp = text[work_exp_start+1:work_exp_end-2]

# # Find the start and end indices of the project section
# project_start = work_exp_end
# project_end = len(text)
# for i, token in enumerate(doc):
#     if token.text.lower() == "projects" or token.text.lower() == "project experience" or token.text.lower() == "project":
#         project_start = token.idx
#     elif token.text.lower() == "skills" or token.text.lower() == "activities" or token.text.lower() == "awards" or token.text.lower() == "achievements":
#         project_end = token.idx
#         break

workEX = ["work experience", "work", "experience", "professional experience", "relevant experience", "work history"]
projectEX = ["projects", "project experience", "project", "project history", "project details"]
workEXend = ["education", "educational background", "education background", "education history", "skills", "activities", "awards", "achievements", "research", "publications", "certifications", "hobbies", "interests", "languages", "references"] + projectEX
projectEXend = ["skills", "activities", "awards", "achievements", "research", "publications", "certifications", "hobbies", "interests", "languages", "references"] + workEX

work_exp_start = 0
work_exp_end = len(text)
for i, token in enumerate(doc):
    if token.text.lower() in workEX:
        work_exp_start = token.idx
    elif token.text.lower() in workEXend:
        work_exp_end = token.idx
        break

project_start = 0
for i, token in enumerate(doc):
    if token.text.lower() in projectEX:
        project_start = token.idx
    elif token.text.lower() in projectEXend:
        project_end = token.idx
        break

work = text[work_exp_start+1:work_exp_end-2]

# Extract the contents of the project section
project = text[project_start+1:project_end-2]

print("Work Experience:\n", work)
print("Project Experience:\n", project)


