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

# --- Piston API configuration ---
PISTON_API = "https://emkc.org/api/v2/piston/execute"

def evaluate_with_piston(source_code: str, stdin: str = "") -> dict:
    payload = {
        "language": "python3",
        "version": "3.10.0",
        "files": [{"name": "main.py", "content": source_code}],
        "stdin": stdin
    }
    res = requests.post(PISTON_API, json=payload)
    if not res.ok:
        return {"error": f"Piston request failed: {res.status_code}"}
    return res.json()

def run_user_function(user_code: str, input_line: str) -> str:
    match = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", user_code)
    if not match:
        return "❌ Could not find function definition."
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
            except:
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
            except:
                return f"❌ Could not parse positional value '{part}'"
        call_str = f"{fn_name}({', '.join(repr(v) for v in vals)})"

    wrapper = f"""
{user_code}
try:
    result = {call_str}
    print(result)
except Exception as e:
    print('RUNTIME_ERROR:', e)
"""
    exec_res = evaluate_with_piston(wrapper)
    if exec_res.get("error"):
        return f"❌ Execution Error: {exec_res['error']}"
    stdout = exec_res.get("run", {}).get("stdout", "")
    stderr = exec_res.get("run", {}).get("stderr", "")
    if stderr:
        return f"❌ Runtime Error: {stderr.strip()}"
    if "RUNTIME_ERROR:" in stdout:
        return f"❌ Runtime Exception: {stdout.split('RUNTIME_ERROR:')[1].strip()}"
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

# Complexity mapping helper
def complexity_rank(comp: str) -> int:
    # maps "O(n)"->1, "O(n^2)"->2, etc.
    if "n^3" in comp: return 3
    if "n^2" in comp: return 2
    if "n" in comp: return 1
    return 0


def detect_complexity(user_code: str) -> str:
    class ForDepth(ast.NodeVisitor):
        def __init__(self):
            self.max_depth = 0
            self.current = 0
        def visit_For(self, node):
            self.current += 1
            self.max_depth = max(self.max_depth, self.current)
            self.generic_visit(node)
            self.current -= 1
    try:
        tree = ast.parse(user_code)
        fd = ForDepth()
        fd.visit(tree)
        if fd.max_depth >= 3:
            return 'O(n^3)'
        elif fd.max_depth == 2:
            return 'O(n^2)'
        else:
            return 'O(n)'
    except Exception:
        return 'Unknown'



def rag_answer(query: str, user_code: str = ""):
    model, index, problems = load_rag_resources()
    vec = model.encode([query], convert_to_numpy=True)
    _, I = index.search(vec, 5)
    retrieved = [problems[i] for i in I[0]]


    user_comp = detect_complexity(user_code)
    st.write("🔎 Detected Complexity:", user_comp)

    suggestion = "✅ Your solution appears optimal; no improvement needed."
    if retrieved:
        best = retrieved[0]
        opt_code = best.get('optimal_code', '').strip()
        opt_comp = best.get('time_complexity', '')

        if complexity_rank(user_comp) > complexity_rank(opt_comp) and opt_code:
            approach = best.get('approach', '')
            suggestion = (
                "💡 Your solution is suboptimal. Recommended approach: " + approach +
                "\nHere's optimal code (" + opt_comp + "):\n```python\n" + opt_code + "\n```"
            )
    st.write("🔍 Similar Problems:")
    for p in retrieved:
        st.markdown(f"- {p['title']}")
    return {'suggestion': suggestion}


def main():
    if "test_cases" not in st.session_state:
        st.session_state.test_cases = [{"input": "", "expected": ""}]
    if st.button("➕ Add Test Case"):
        st.session_state.test_cases.append({"input": "", "expected": ""})

    st.title("💡 Code Quality Evaluator + RAG Assistant")
    with st.form("eval_form"):
        st.subheader("📘 Problem Statement")
        problem = st.text_area("", height=150)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🧠 Your Python Code")
            code = st_ace(language="python", theme="github", height=300, key="ace")
        with c2:
            st.subheader("🧪 Test Cases")
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
                        st.success("✅ Passed")
                    else:
                        st.error(f"❌ Expected: {tc['expected']}")
                except:
                    st.error("❌ Invalid output or expected format.")

        with st.spinner("🔍 Generating RAG suggestions..."):
            rag_out = rag_answer(problem, code)

        st.subheader("🛠️ Suggested Improvement Based on RAG")
        st.markdown(rag_out['suggestion'])

if __name__ == "__main__":
    main()
