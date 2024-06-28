import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load the MarianMT model and tokenizer for English to Arabic
model_name = 'Helsinki-NLP/opus-mt-en-ar'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)


def translate_to_arabic(text):
    # Split the input text into lines
    lines = text.split('\n')

    # Translate each line individually
    translated_lines = []
    for line in lines:
        if line.strip() == "":  # Skip empty lines
            translated_lines.append("")
            continue
        inputs = tokenizer(line, return_tensors="pt", truncation=True, padding=True)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_lines.append(translated_text)

    return translated_lines


# Set the title of the Streamlit app
st.title("English to Arabic  Translator")

# Add a text area for user input
text = st.text_area("Enter English text to translate")

# Add a button to trigger the translation
if st.button("Translate"):
    if text:
        translated_lines = translate_to_arabic(text)
        st.write("Translated text:")
        for line in translated_lines:
            st.write(line)
    else:
        st.write("Please enter some text to translate.")
