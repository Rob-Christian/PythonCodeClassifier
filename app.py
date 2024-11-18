import streamlit as st
from transformers import pipeline

# Set up the Streamlit page title
st.title('Python Code Explainer')

# Provide instructions
st.write("""
    **Instructions**: Paste your Python code into the box below, 
    and click the 'Explain Code' button to get an explanation.
""")

# Create a text area for user input (Python code)
code_input = st.text_area("Enter your Python code here", height=300)

# Load the language model for code explanation (using Hugging Face)
llm_explainer = pipeline("text2text-generation", model="Salesforce/codegen-350M-mono")

# Function to generate explanation for the code
def explain_code(code):
    prompt = f"Explain this Python code:\n\n{code}"
    response = llm_explainer(prompt, max_length=200)[0]['generated_text']
    return response

# Button to trigger explanation
if st.button('Explain Code'):
    if code_input.strip() == "":
        st.error("Please enter some Python code to explain.")
    else:
        with st.spinner('Explaining code...'):
            explanation = explain_code(code_input)
        st.write("### Explanation:")
        st.write(explanation)