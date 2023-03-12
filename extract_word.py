import PyPDF2
import spacy
import re
import pickle
import keras
import numpy as np

pdf_file = open("RISHABH's_RESUME.docx.pdf", 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)

for page_num in range(len(pdf_reader.pages)):
    
    page = pdf_reader.pages[page_num]
    text = page.extract_text()
    text = text.lower()



# load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

def all_words(text):
    doc = nlp(text)
    # Find the start and end indices of the work experience section
    work_exp_start = 0
    work_exp_end = len(text)
    for i, token in enumerate(doc):
        if token.text.lower() == "work experience" or token.text.lower() == "work" or token.text.lower() == "experience" or token.text.lower() == "professional experience" or token.text.lower() == "relevant experience":
            work_exp_start = token.idx
        elif token.text.lower() == "education" or token.text.lower() == "educational background" or token.text.lower() == "education background":
            work_exp_end = token.idx
            break
    work_exp = text[work_exp_start+1:work_exp_end-2]
    # Find the start and end indices of the project section
    project_start = work_exp_end
    project_end = len(text)
    for i, token in enumerate(doc):
        if token.text.lower() == "projects" or token.text.lower() == "project experience" or token.text.lower() == "project":
            project_start = token.idx
        elif token.text.lower() == "skills" or token.text.lower() == "activities" or token.text.lower() == "awards" or token.text.lower() == "achievements":
            project_end = token.idx
            break
    work = text[work_exp_start+1:work_exp_end-2]
    # Extract the contents of the project section
    project = text[project_start+1:project_end-2]

    work = re.sub(r'[^a-zA-Z0-9]', ' ', work)
    project = re.sub(r'[^a-zA-Z0-9]', ' ', project)

    work = work.split('\n')
    project = project.split('\n')

    all = work + project
    return all
    


def makePredictions(text):
    vocab = pickle.load(open("machine_learning/models/vocab.pkl", "rb"))
    model = keras.models.load_model("machine_learning/models/model_for_inference")
    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"]
    )
    text = text.lower()
    text = [text]
    text = np.array(text)
    text = text.reshape(1, -1)
    predictions = model.predict(text)
    top_3 = np.argsort(predictions[0])[-3:][::-1]
    return [vocab[i] for i in top_3]

def predict(text):
    all = all_words(text)
    skills = []
    for i in all:
        skills.append(makePredictions(i))
    return skills

print(predict(text))
