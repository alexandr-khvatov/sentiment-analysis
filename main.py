import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf

from text_preprocessing import preprocessing
from vectorizer import vectorize_sentence

MODEL_PATH = 'saved/model/Glove-Navec_BiLSTM_remove_stopwords'


def make_vectorize(raws):
    return np.array(list(map(lambda sentence: vectorize_sentence(preprocessing(sentence)), raws)))


@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

st.title('Toxic classification')
st.info("Glove(Navec) embeddings and BiLSTM model")
user_input_raw = st.text_area("Enter your text here", "ты чего берега попутал? Это правый или левый берег реки", height=50)
predict_btt = st.button("PREDICT")
if predict_btt:
    x = make_vectorize([user_input_raw])
    pred = model.predict(x).flatten()
    st.write(("Toxic:  {0:.2f} %".format(100 * pred[0])))
    class_label = ["Toxic", "Neutral"]
    prob_list = [pred[0] * 100, 100 - (pred[0] * 100)]
    prob_dict = {"Toxic/Neutral": class_label, "Likelihood": prob_list}
    df_prob = pd.DataFrame(prob_dict)
    fig = px.bar(df_prob, x='Toxic/Neutral', y='Likelihood')
    st.plotly_chart(fig, use_container_width=True)