\import streamlit as st
from streamlit_ace import st_ace
import ast
import re
import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist # Import cdist for similarity search

st.set_page_config(page_title="Code Quality Evaluator + RAG", layout="wide")


PISTON_API = "https://emkc.org/api/v2/piston/execute"

def evaluate_with_piston(source_code: str, stdin: str = "") -> dict:
    """Sends code to the Piston API for execution."""
    payload = {
        "language": "python3",
        "version": "3.10.0",
        "files": [{"name": "main.py", "content": source_code}],
        "stdin": stdin
    }
    try:
        res = requests.post(PISTON_API, json=payload, timeout=10)
        res.raise_for_status()
    except requests.RequestException as e:
        return {"error": f"Piston request failed: {e}"}
    return res.json()

def run_user_function(user_code: str, input_line: str) -> str:
    """Parses user code, prepares arguments, and executes it via Piston API."""
    try:
        tree = ast.parse(user_code)
        fn_node = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
        if not fn_node:
            return "❌ Could not find a top-level function definition."
        fn_name = fn_node.name
        params = [arg.arg for arg in fn_node.args.args]
    except Exception:
        return "❌ Error parsing function definition."

    # Prepare arguments from the input string
    if "Input:" in input_line:
        input_line = input_line.split("Input:", 1)[1].strip()
    parts = re.split(r",\s*(?![^\[\](){}]*[\]\)\}])", input_line)

    # Handle keyword arguments vs. positional arguments
    if any("=" in part for part in parts):
        args_dict = {}
        for part in parts:
            if "=" not in part:
                continue
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
            if not part:
                continue
            try:
                vals.append(ast.literal_eval(part))
            except:
                return f"❌ Could not parse positional value '{part}'"
        call_str = f"{fn_name}({', '.join(repr(v) for v in vals)})"

    # Build a wrapper script to execute the function and capture output
    wrapper = user_code.strip() + "\n"
    wrapper += f"try:\n    result = {call_str}\n    print(result)\n"
    wrapper += "except Exception as e:\n    print('RUNTIME_ERROR:', e)\n"

    exec_res = evaluate_with_piston(wrapper)
    if exec_res.get("error"):
        return f"❌ Execution Error: {exec_res['error']}"

    run_info = exec_res.get("run", {})
    stdout = run_info.get("stdout", "")
    stderr = run_info.get("stderr", "")
    if stderr:
        return f"❌ Runtime Error: {stderr.strip()}"
    if "RUNTIME_ERROR:" in stdout:
        return f"❌ Runtime Exception: {stdout.split('RUNTIME_ERROR:')[1].strip()}"
    return stdout.strip() or "⚠️ No output returned."

# --- RAG resources loader (Now using SciPy/NumPy, no FAISS) ---
@st.cache_resource(ttl=3600)
def load_rag_resources():
    """Loads the model and problem data, and pre-computes embeddings."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    with open("rag_minilm_problems.json", "r", encoding="utf-8") as f:
        problems = json.load(f)
    # Pre-compute the embeddings for all problem titles for faster search
    problem_embeddings = model.encode([p['title'] for p in problems])
    return model, problems, problem_embeddings

# --- Helper functions for RAG ---
def complexity_rank(comp: str) -> int:
    """Assigns a numerical rank to complexity strings for comparison."""
    if "n^3" in comp: return 3
    if "n^2" in comp: return 2
    if "n" in comp: return 1
    return 0

def detect_complexity(user_code: str) -> str:
    """Analyzes code with AST to find the maximum loop nesting depth."""
    class LoopDepth(ast.NodeVisitor):
        def __init__(self):
            self.max_depth = 0
            self.current = 0
        def visit_For(self, node): self._enter_loop(node)
        def visit_While(self, node): self._enter_loop(node)
        def _enter_loop(self, node):
            self.current += 1
            self.max_depth = max(self.max_depth, self.current)
            self.generic_visit(node)
            self.current -= 1
    try:
        tree = ast.parse(user_code)
        ld = LoopDepth()
        ld.visit(tree)
        if ld.max_depth >= 3: return 'O(n^3)'
        if ld.max_depth == 2: return 'O(n^2)'
        return 'O(n)'
    except Exception:
        return 'Unknown'

def rag_answer(query: str, user_code: str = ""):
    """Finds similar problems using SciPy and suggests improvements."""
    model, problems, problem_embeddings = load_rag_resources()
    query_vec = model.encode([query])

    # --- SciPy-based similarity search ---
    # Calculate cosine distance between the query and all problem titles
    distances = cdist(query_vec, problem_embeddings, "cosine")[0]
    # Get the indices of the 5 problems with the smallest distance
    top_indices = np.argsort(distances)[:5]
    retrieved = [problems[i] for i in top_indices]
    # --- End of SciPy search ---

    user_comp = detect_complexity(user_code)
    st.write(" Detected Complexity:", user_comp)

    suggestion = " Your solution appears optimal; no improvement needed."
    if retrieved:
        best = retrieved[0]
        opt_code = best.get('optimal_code', '').strip()
        opt_comp = best.get('time_complexity', '')
        if complexity_rank(user_comp) > complexity_rank(opt_comp) and opt_code:
            approach = best.get('approach', '')
            suggestion = (
                f" Your solution is suboptimal. Recommended approach: {approach}\n"
                f"Here's an optimal code snippet ({opt_comp}):\n```python\n{opt_code}\n```"
            )
            
    st.write(" Similar Problems Found:")
    for p in retrieved:
        st.markdown(f"- {p['title']}")
    return {'suggestion': suggestion}

# --- Main Streamlit App UI ---
def main():
    st.title("Code Quality Evaluator + RAG Assistant")
    if "test_cases" not in st.session_state:
        st.session_state.test_cases = [{"input": "", "expected": ""}]
 
    if st.button(" Add Test Case"):
        st.session_state.test_cases.append({"input": "", "expected": ""})

    with st.form("eval_form"):
        st.subheader(" Problem Statement")
        problem = st.text_area("Describe the problem you are trying to solve.", height=150)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader(" Your Python Code")
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

        submit = st.form_submit_button(" Evaluate")

    if submit and problem.strip():
        st.markdown("---")
        st.subheader(" Test Case Results")
        all_passed = True
        for i, tc in enumerate(st.session_state.test_cases):
            st.markdown(f"**Test Case {i+1}**")
            output = run_user_function(code, tc["input"])
            st.text(f"🔧 Output: {output}")
            if output.startswith("❌"):
                st.error("Execution error; skipping suggestions.")
                all_passed = False
            elif tc.get("expected"):
                try:
                    if ast.literal_eval(output.strip()) == ast.literal_eval(tc["expected"].strip()):
                        st.success(" Passed")
                    else:
                        st.error(f"❌ Failed. Expected: {tc['expected']}")
                        all_passed = False
                except:
                    st.error("❌ Invalid output or expected format.")
                    all_passed = False
        
        # Only run RAG if the code is valid and test cases passed
        if all_passed:
            with st.spinner(" Generating RAG suggestions..."):
                rag_out = rag_answer(problem, code)
            st.subheader("Suggested Improvement Based on RAG")
            st.markdown(rag_out['suggestion'])

if __name__ == "__main__":
    main()
