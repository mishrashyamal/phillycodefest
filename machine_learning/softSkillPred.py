import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow import keras
from ast import literal_eval
import tensorflow as tf
import pickle


df = pd.read_csv('../data/soft_skills.csv')


df["soft_skills"] = df["soft_skills"].apply(
    lambda x: literal_eval(x)
)

class_counts = df["soft_skills"].value_counts()
from sklearn.model_selection import train_test_split

valid_classes = class_counts[class_counts >= 2].index.tolist()

df_filtered = df[df["soft_skills"].isin(valid_classes)]

train_df, test_df = train_test_split(
    df_filtered,
    test_size= 0.2,
    random_state=42,
    stratify=df_filtered["soft_skills"].values
)

terms = tf.ragged.constant(train_df["soft_skills"].values)
lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
lookup.adapt(terms)
vocab = lookup.get_vocabulary()


def invert_multi_hot(encoded_labels):
    """Reverse a single multi-hot encoded label to a tuple of vocab terms."""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(vocab, hot_indices)

max_seqlen = 150
batch_size = 128
padding_token = "<pad>"
auto = tf.data.AUTOTUNE


def make_dataset(dataframe, is_train=True):
    labels = tf.ragged.constant(dataframe["soft_skills"].values)
    label_binarized = lookup(labels).numpy()
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["text"].values, label_binarized)
    )
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    return dataset.batch(batch_size)

train_dataset = make_dataset(train_df, is_train=True)
validation_dataset = make_dataset(test_df, is_train=False)
test_dataset = make_dataset(test_df, is_train=False)

vocabulary = set()
train_df["text"].str.lower().str.split().apply(vocabulary.update)
vocabulary_size = len(vocabulary)

text_vectorizer = layers.TextVectorization(
    max_tokens=vocabulary_size, ngrams=2, output_mode="tf_idf"
)

with tf.device("/CPU:0"):
    text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

train_dataset = train_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)
validation_dataset = validation_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)
test_dataset = test_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)

def make_model():
    shallow_mlp_model = keras.Sequential(
        [
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(lookup.vocabulary_size(), activation="sigmoid"),
        ]  
    )
    return shallow_mlp_model

epochs = 20
shallow_mlp_model = make_model()
shallow_mlp_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"]
)

history = shallow_mlp_model.fit(
    train_dataset, validation_data=validation_dataset, epochs=epochs
)

model_for_inference = keras.Sequential([text_vectorizer, shallow_mlp_model])

##############################################################################################################
## To make predictions
# import pickle
# import keras
# import numpy as np

def makePredictions(text):
    vocab = pickle.load(open("models/vocab.pkl", "rb"))
    model = keras.models.load_model("models/model_for_inference")
    predictions = model.predict([text])
    top_3 = np.argsort(predictions[0])[-3:][::-1]
    return [vocab[i] for i in top_3]
print(makePredictions("I am a good team player and I am a good communicator")) 