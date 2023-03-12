import pickle
import keras
import numpy as np

def makePredictions(text):
    vocab = pickle.load(open("models/vocab.pkl", "rb"))
    model = keras.models.load_model("models/model_for_inference")
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

data = [
    "My name is John and I am a software engineer. I have experience in Python, Java, and C++. I am also familiar with the following frameworks: Tensorflow, Keras, and PyTorch. I am looking for a job in the field of data science.",
    "Working with data is my passion. I ve experience in data analysis, data visualization, and machine learning. I am also familiar with the following tools: Tableau, Power BI, and Qlik. I am looking for a job in the field of data science."
]   

for text in data:
    print(makePredictions(text))