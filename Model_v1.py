import time
import streamlit as st
import torch
import re
import subprocess
import sys
import asyncio
from transformers import RobertaTokenizer, T5ForConditionalGeneration

# Set Streamlit page configuration
st.set_page_config(page_title="TestGenML", layout="wide")

# Load Model & Tokenizer
MODEL_PATH = r"C:\\Users\\User\\PycharmProjects\\MLproject\\trained_codet5v5"

@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model()

# Function to generate test cases
async def generate_test_case(model, tokenizer, function_code):
    model.eval()
    input_text = "Generate a unit test for:\n" + function_code
    input_tokens = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = input_tokens["input_ids"].to(device)
    attention_mask = input_tokens["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text

# Function to clean and normalize code
async def clean_code(code):
    return re.sub(r'[\u00A0\u200B\u200E\u200F\u2028\u2029]', ' ', code)

# Function to modify test cases and add exception handling
async def add_exception_handling(code):
    lines = code.split("\n")
    modified_lines = []
    function_names = []

    for line in lines:
        match = re.match(r"def\s+(\w+)\(", line)
        if match:
            function_name = match.group(1)
            function_names.append(function_name)
            modified_lines.append(line)
        else:
            modified_lines.append(line)

    # Add exception handling for each test function call
    modified_lines.append("\n# Automatically call the test functions")
    for func in function_names:
        modified_lines.append(f"""
try:
    {func}()
    print('PASSED: {func}')
except AssertionError:
    print('FAILED: {func} (Assertion Failed)')
except Exception as e:
    print(f'FAILED: {func} (Error: {{str(e)}})')
""")
    return "\n".join(modified_lines)

# Function to run generated test cases and count passed tests
async def run_test_cases(function_code, test_code):
    passed_count = 0
    failed_count = 0
    output_log = []

    # Combine function code and test code
    complete_code = await clean_code(function_code + "\n\n" + test_code)

    # Save the complete code to a temporary Python file with UTF-8 encoding
    with open("temp_test.py", "w", encoding="utf-8") as f:
        f.write(complete_code)

    try:
        # Run the test file and capture output
        result = subprocess.run([sys.executable, "temp_test.py"], capture_output=True, text=True)
        output = result.stdout + result.stderr
        output_log.append(output)

        # Count passed and failed tests
        passed_count = output.count("PASSED:")
        failed_count = output.count("FAILED:")

    except Exception as e:
        output_log.append(f"Error executing test cases: {str(e)}")

    return passed_count, failed_count, "\n".join(output_log)

# Initialize session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title Section
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>TestGenML</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #777;'>Type a function, and I'll generate test cases!</h5>", unsafe_allow_html=True)
st.markdown("---")

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.code(chat["input"], language="python")
    with st.chat_message("assistant"):
        st.code(chat["output"], language="python")

# User Input
function_code = st.chat_input("Enter your Python function here...")

if function_code:
    # Generate test case
    start_time = time.time()
    generated_test = asyncio.run(generate_test_case(model, tokenizer, function_code))
    response_time = time.time() - start_time
    updated_test_cases = asyncio.run(add_exception_handling(generated_test))

    # Run the test cases and get the result
    passed, failed, test_output = asyncio.run(run_test_cases(function_code, updated_test_cases))

    # Create a summary of test results
    summary = f"✅ Passed: {passed} | ❌ Failed: {failed} "
    response = f"\n\n{summary}\n\n{test_output}"

    # Store in session state
    st.session_state.chat_history.append({"input": function_code, "output": response})

    # Display immediately
    with st.chat_message("user"):
        st.code(function_code, language="python")
    with st.chat_message("assistant"):
        st.code(response, language="python")
