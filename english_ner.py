import streamlit as st
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForTokenClassification.from_pretrained("./english_ner_model")
    tokenizer = AutoTokenizer.from_pretrained("./english_ner_model")
    return model, tokenizer

def format_ner_output_html(text, entities):
    """
    Construct an HTML string where each entity's label is highlighted.
    The output will be similar to:
    "Barack Hussein Obama II <span style='...'>PERSON</span> (born August 4, 1961) is an ..."
    """
    # Sort entities by start offset
    entities = sorted(entities, key=lambda x: x["start"])
    last = 0
    parts = []
    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        label = ent["label"]
        # Append text before the entity
        parts.append(text[last:start])
        # Extract entity text
        entity_text = text[start:end]
        # Append entity text followed by a highlighted label
        parts.append(
            f"{entity_text} <span style='background-color: #f9c74f; color: black; padding: 2px 4px; border-radius: 4px;'>{label}</span>"
        )
        last = end
    parts.append(text[last:])
    return "".join(parts)

st.title("English NER Model Visualization")
model, tokenizer = load_model()

text = st.text_area(
    "Enter a sentence:",
    "Barack Hussein Obama II (born August 4, 1961) is an American attorney and politician who served as the 44th President of the United States from January 20, 2009 to January 20, 2017. A member of the Democratic Party, he was the first African American to serve as president. He was previously a United States Senator from Illinois and a member of the Illinois State Senate."
)

if st.button("Predict"):
    # Tokenize text with character offsets
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = inputs["offset_mapping"][0].tolist()
    
    # Get model predictions (excluding offset mapping from inputs)
    with torch.no_grad():
        logits = model(**{k: v for k, v in inputs.items() if k != "offset_mapping"}).logits
    predicted_ids = logits.argmax(dim=-1)[0]
    
    # Process entities assuming BIO scheme
    entities = []
    current_entity = None
    for token_id, (start, end) in zip(predicted_ids, offsets):
        label = model.config.id2label[token_id.item()]
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "start": start,
                "end": end,
                "label": label.split("-")[1]
            }
        elif label.startswith("I-") and current_entity:
            current_entity["end"] = end
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    if current_entity:
        entities.append(current_entity)
    
    # Format the text as HTML with highlighted entity labels
    formatted_html = format_ner_output_html(text, entities)
    st.markdown(formatted_html, unsafe_allow_html=True)
