import json, sys, urllib.request

base, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
PROMPTS = [
    ("math", "What is 17*23+45? Answer with just the number."),
    ("code", "Write a Python function to reverse a linked list."),
    ("factual", "Name the capital of Australia and its approximate population."),
    ("reasoning", "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Answer yes or no with one sentence of explanation."),
    ("edit", "Fix the bug in this code: def add(a, b): return a - b"),
    ("explain", "Explain in 3 sentences how speculative decoding works."),
]
res = {}
for name, p in PROMPTS:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": p}],
                       "max_tokens": 250, "temperature": 0}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=240))
    m = r["choices"][0]["message"]
    # qwen3 reasoning parser splits early tokens into reasoning_content
    res[name] = {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or ""}
json.dump(res, open(out, "w"), indent=1)
print("captured", len(res), "prompts ->", out)
