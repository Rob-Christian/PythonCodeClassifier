from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

# Load the CodeT5 model and tokenizer
@st.cache_resource
def load_codet5_model():
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base")
    return tokenizer, model

tokenizer, model = load_codet5_model()

# Function to explain a single line of code using CodeT5
def explain_code_codet5(line):
    input_ids = tokenizer(f"explain code: {line}", return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=100, temperature=0.7, repetition_penalty=1.2)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Clean the explanation to remove redundant sentences
def clean_explanation(text):
    sentences = text.split(". ")
    seen = set()
    cleaned = []
    for sentence in sentences:
        if sentence not in seen:
            cleaned.append(sentence)
            seen.add(sentence)
    return ". ".join(cleaned)

# Function to process the code line by line
def explain_code_line_by_line(code):
    lines = [line.strip() for line in code.strip().split("\n") if line.strip()]
    explanations = []
    
    for line in lines:
        try:
            explanation = explain_code_codet5(line)
            cleaned_explanation = clean_explanation(explanation)
            explanations.append(f"**Code:** `{line}`\n**Explanation:** {cleaned_explanation}")
        except Exception as e:
            explanations.append(f"**Code:** `{line}`\n**Explanation:** Unable to process. Error: {str(e)}")
    
    return explanations

# Streamlit UI
st.title("Python Code Explainer")
st.write("Provide a Python code snippet, and this app will explain each line using CodeT5.")

# Input area for the user to provide code
user_code = st.text_area("Enter your Python code below:", height=200)

# Process and display explanations
if st.button("Explain Code"):
    if user_code.strip():
        st.write("### Line-by-Line Explanation:")
        explanations = explain_code_line_by_line(user_code)
        for explanation in explanations:
            st.markdown(explanation)
    else:
        st.warning("Please enter some Python code to explain.")
