import streamlit as st
from span_marker import SpanMarkerModel

# Cache the model loading to avoid reloading on every prediction
@st.cache_resource
def load_model():
    return SpanMarkerModel.from_pretrained("tomaarsen/span-marker-roberta-tagalog-base-tlunified")

def highlight_entities(text, entities):
    # Use the correct keys: char_start_index and char_end_index
    sorted_entities = sorted(entities, key=lambda x: x['char_end_index'], reverse=True)
    text_list = list(text)
    
    for entity in sorted_entities:
        start = entity['char_start_index']
        end = entity['char_end_index']
        label = entity['label']
        text_list.insert(end, f"</mark>")
        text_list.insert(start, f"<mark style='background-color: #90EE90; padding: 2px; border-radius: 3px;'>{label}</mark> ")
    
    return "".join(text_list)

# Streamlit app
st.title("Tagalog NER with SpanMarker")
st.write("Enter Tagalog text to identify entities (PERSON, LOC, ORG, etc.)")

# Load model once
model = load_model()

# User input
text = st.text_area("Input text:", "Si Juan dela Cruz ay nanirahan sa Maynila.")

if st.button("Analyze"):
    if text.strip():
        # Get predictions
        entities = model.predict(text)
        
        # Display raw entities
        st.subheader("Identified Entities")
        st.json(entities)
        
        # Display highlighted text
        st.subheader("Visualized Entities")
        highlighted = highlight_entities(text, entities)
        st.markdown(highlighted, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to analyze")