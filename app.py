import streamlit as st
from transformers import pipeline
import torch

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load the model
st.write("Loading the model...")
llm_explainer = pipeline("text2text-generation", model="t5-base", device=device)

st.title("Python Code Explainer")

# User input for Python code
code_input = st.text_area("Enter your Python code here:", height=300)

# Function to explain each line of code
def explain_code_line_by_line(code):
    # Split the code into lines
    lines = code.strip().split("\n")
    explanations = []
    
    for line in lines:
        if line.strip():  # Skip empty lines
            prompt = f"Explain the following Python code line in detail:\n\n{line}"
            try:
                response = llm_explainer(prompt, max_length=100, truncation=True)[0]["generated_text"]
                explanations.append(f"**Code:** `{line}`\n**Explanation:** {response}")
            except Exception as e:
                explanations.append(f"**Code:** `{line}`\n**Explanation:** Unable to process. Error: {str(e)}")
    return explanations

# Display explanation
if st.button("Explain Code"):
    if code_input.strip():
        st.write("### Line-by-Line Explanation:")
        explanations = explain_code_line_by_line(code_input)
        for explanation in explanations:
            st.markdown(explanation)
    else:
        st.warning("Please enter Python code to get an explanation.")
