import streamlit as st
from streamlit_ace import st_ace
import subprocess
import ast
import re
import json
import time
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter

st.set_page_config(page_title="Code Quality Evaluator + RAG", layout="wide")


JUDGE0_API = "https://judge0-ce.p.rapidapi.com/submissions"
JUDGE0_HEADERS = {
    "x-rapidapi-host": "judge0-ce.p.rapidapi.com",
    "x-rapidapi-key": st.secrets.get("judge0_key", ""),
    "content-type": "application/json"
}

def evaluate_with_judge0(source_code: str, language_id: int = 71, stdin: str = "") -> dict:
    payload = {"source_code": source_code, "language_id": language_id, "stdin": stdin}
    res = requests.post(JUDGE0_API, headers=JUDGE0_HEADERS, json=payload)
    if not res.ok:
        return {"error": f"Judge0 POST failed: {res.status_code}"}
    token = res.json().get("token")
    if not token:
        return {"error": "No token returned from Judge0"}
    start = time.time()
    while True:
        status = requests.get(f"{JUDGE0_API}/{token}", headers=JUDGE0_HEADERS).json()
        st_id = status.get("status", {}).get("id")
        if st_id in (1, 2):
            if time.time() - start > 10:
                return {"error": "Judge0 timeout"}
            time.sleep(0.5)
            continue
        return status

def run_user_function(user_code: str, input_line: str) -> str:
    syntax_status = evaluate_with_judge0(user_code, language_id=71)
    if syntax_status.get("error"):
        return f"❌ Judge0 Error: {syntax_status['error']}"
    if syntax_status.get("compile_output"):
        return f"❌ Syntax Error:\n{syntax_status['compile_output']}"

    match = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", user_code)
    if not match:
        return " Could not find function definition."
    fn_name = match.group(1)
    params = [p.strip() for p in match.group(2).split(',') if p.strip()]

    if "Input:" in input_line:
        input_line = input_line.split("Input:", 1)[1].strip()
    parts = re.split(r",\s*(?![^\[\](){}]*[\]\)\}])", input_line)

    if any("=" in part for part in parts):
        args_dict = {}
        for part in parts:
            if "=" not in part: continue
            k, v = part.split("=", 1)
            try:
                args_dict[k.strip()] = ast.literal_eval(v.strip())
            except Exception:
                return f"❌ Could not parse value for '{k.strip()}'"
        args_list = [repr(args_dict.get(p)) for p in params]
        call_str = f"{fn_name}({', '.join(args_list)})"
    else:
        vals = []
        for part in parts:
            part = part.strip()
            if not part: continue
            try:
                vals.append(ast.literal_eval(part))
            except Exception:
                return f"❌ Could not parse positional value '{part}'"
        call_str = f"{fn_name}({', '.join(repr(v) for v in vals)})"

    wrapper = f"""
import json
{user_code}
try:
    result = {call_str}
    print(result)
except Exception as e:
    print("RUNTIME_ERROR:", e)
"""
    exec_status = evaluate_with_judge0(wrapper, language_id=71)
    if exec_status.get("error"):
        return f"❌ Judge0 Execution Error: {exec_status['error']}"
    stdout = exec_status.get("stdout") or ""
    stderr = exec_status.get("stderr") or ""
    if "RUNTIME_ERROR:" in stdout:
        return f"❌ Runtime Exception: {stdout.split('RUNTIME_ERROR:')[1].strip()}"
    if stderr:
        return f"❌ Runtime Error: {stderr}"
    return stdout.strip() or "⚠️ No output returned."


def load_rag_resources():

    return _load_rag_resources()

@st.cache_resource
def _load_rag_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("rag_minilm_index.faiss")
    with open("rag_minilm_problems.json", "r", encoding="utf-8") as f:
        problems = json.load(f)
    return model, index, problems


def rag_answer(query: str, user_code: str = ""):
    model, index, problems = load_rag_resources()
    vec = model.encode([query], convert_to_numpy=True)
    _, I = index.search(vec, 5)
    retrieved = [problems[i] for i in I[0]]

    approaches = [p.get('approach') for p in retrieved if p.get('approach')]
    dominant = Counter(approaches).most_common(1)[0][0] if approaches else 'Other'

    st.write("🔎 Detected Approach:", dominant)


    suggestion = ''
    if dominant.lower() in ['brute force', 'brute']:
        suggestion = (
            "💡 Your solution uses a brute-force approach (O(n³)). "
            "Consider using the optimal 'Two Pointers' technique to achieve O(n²) time."
        )
    else:
        suggestion = "✅ Your solution appears optimal; no improvement needed."

 
    titles = [p['title'] for p in retrieved]

    return {'retrieved_titles': titles, 'approach': dominant, 'suggestion': suggestion}


def main():
    if "test_cases" not in st.session_state:
        st.session_state.test_cases = [{"input": "", "expected": ""}]
    if st.button(" Add Test Case"):
        st.session_state.test_cases.append({"input": "", "expected": ""})

    st.title(" Code Quality Evaluator + RAG Assistant")
    with st.form("eval_form"):
        st.subheader("📘 Problem Statement")
        problem = st.text_area("", height=150)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🧠 Your Python Code")
            code = st_ace(language="python", theme="github", height=300, key="ace")
        with c2:
            st.subheader(" Test Cases")
            updated = []
            for i, tc in enumerate(st.session_state.test_cases):
                st.markdown(f"**Test Case {i+1}**")
                inp = st.text_area(f"Input {i}", tc["input"], key=f"in{i}")
                exp = st.text_area(f"Expected {i}", tc["expected"], key=f"ex{i}")
                updated.append({"input": inp, "expected": exp})
            st.session_state.test_cases = updated
        submit = st.form_submit_button("🔍 Evaluate")

    if submit and problem.strip():
        st.markdown("---")
        st.subheader("🧪 Test Case Results")
        for i, tc in enumerate(st.session_state.test_cases):
            st.markdown(f"**Test Case {i+1}**")
            output = run_user_function(code, tc["input"])
            st.text(f"🔧 Output: {output}")
            if output.startswith("❌"):
                st.error("Execution error; skipping suggestions.")
            elif tc.get("expected"):
                try:
                    if ast.literal_eval(output.strip()) == ast.literal_eval(tc["expected"].strip()):
                        st.success("✅Passed")
                    else:
                        st.error(f" Expected: {tc['expected']}")
                except Exception:
                    st.error(" Invalid output or expected format.")

        with st.spinner(" Generating RAG suggestions..."):
            rag_out = rag_answer(problem, code)

        st.subheader(" Suggested Improvement Based on RAG")
        st.markdown(rag_out['suggestion'])
        st.subheader(" Similar Problems")
        for title in rag_out['retrieved_titles']:
            st.markdown(f"- {title}")

if __name__ == "__main__":
    main()