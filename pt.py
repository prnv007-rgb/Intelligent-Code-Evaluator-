# parser_test.py

from test import run_user_function

# A minimal threeSum implementation for testing
USER_CODE = """
def threeSum(nums):
    nums.sort()
    res = []
    for i, a in enumerate(nums):
        if a > 0:
            break
        if i > 0 and a == nums[i - 1]:
            continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            three = a + nums[l] + nums[r]
            if three > 0:
                r -= 1
            elif three < 0:
                l += 1
            else:
                res.append([a, nums[l], nums[r]])
                l += 1
                r -= 1
                while l < r and nums[l] == nums[l - 1]:
                    l += 1
    return res
"""

TEST_INPUTS = [
    "nums=[-1,0,1,2,-1,-4]",       # named
    "[-1,0,1,2,-1,-4]",            # positional
    "Input: nums=[-1,0,1,2,-1,-4]",# named with prefix
    "Input: [-1,0,1,2,-1,-4]"      # positional with prefix
]

for inp in TEST_INPUTS:
    output = run_user_function(USER_CODE, inp)
    print(f"INPUT → {inp!r}\nOUTPUT → {output}\n{'-'*50}")
