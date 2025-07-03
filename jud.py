import requests
import time

JUDGE0_API = "https://judge0-ce.p.rapidapi.com"
HEADERS = {
    "x-rapidapi-host": "judge0-ce.p.rapidapi.com",
    "x-rapidapi-key": "6ef5dc0a6dmshb80e9b4e3b6095ap18b321jsn7bbee28b43cd",  # ⬅️ Replace this
    "content-type": "application/json"
}

def run_code(source_code, stdin, language_id=71):  # Python3 = 71
    # Step 1: Submit the code
    submission = requests.post(
        f"{JUDGE0_API}/submissions?base64_encoded=false&wait=false",
        headers=HEADERS,
        json={
            "source_code": source_code,
            "language_id": language_id,
            "stdin": stdin,
            "redirect_stderr_to_stdout": True
        }
    )
    token = submission.json()["token"]

    # Step 2: Poll for result
    while True:
        result = requests.get(f"{JUDGE0_API}/submissions/{token}", headers=HEADERS).json()
        status_id = result["status"]["id"]
        if status_id in [1, 2]:  # In Queue / Processing
            time.sleep(1)
        else:
            return {
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "status": result["status"]["description"]
            }
user_code = """
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                print([i, j])

target = int(input())
two_sum(nums, target)
"""

input_data = "2 7 11 15\n9"
result = run_code(user_code, input_data)

print("Status:", result["status"])
print("Output:\n", result["stdout"])
print("Errors:\n", result["stderr"])
