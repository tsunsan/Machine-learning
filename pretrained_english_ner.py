import spacy_streamlit

models = ["en_core_web_trf"]
default_text = "Sundar Pichai is the CEO of Google."
spacy_streamlit.visualize(models, default_text)