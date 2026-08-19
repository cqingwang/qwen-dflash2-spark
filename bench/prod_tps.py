import json, time, urllib.request
URL = "http://192.168.1.205:8003/v1/completions"
def bench(n, stream=False):
    payload = json.dumps({
        "model": "qwen3.8-27b-tp2",
        "prompt": "Write a detailed technical essay about memory bandwidth constraints in modern AI accelerators.",
        "max_tokens": n, "temperature": 0.7, "ignore_eos": True, "stream": stream
    }).encode()
    req = urllib.request.Request(URL, payload, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=180)
    if stream:
        first = None; ntok = 0
        for line in resp:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                d = json.loads(line[5:])
                if d.get("choices"):
                    if first is None: first = time.perf_counter()
                    if d["choices"][0].get("text"): ntok += 1
        t1 = time.perf_counter()
        print(f"stream: TTFT {(first-t0)*1e3:.0f}ms | {ntok} tok in {t1-first:.2f}s -> {ntok/(t1-first):.1f} tok/s decode")
    else:
        d = json.load(resp)
        dt = time.perf_counter() - t0
        ct = d["usage"]["completion_tokens"]
        print(f"non-stream: {ct} tok in {dt:.2f}s -> {ct/dt:.1f} tok/s (incl prefill)")
bench(256, stream=True)
bench(256, stream=True)
bench(256)
