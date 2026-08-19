import json, os, shutil, torch
from safetensors.torch import load_file, save_file

SRC = os.environ.get("SRC", "/work/unsloth-nvfp4")
DST = os.environ.get("DST", "/work/unsloth-nvfp4-bf16head")

if not os.path.isdir(DST):
    shutil.copytree(SRC, DST, copy_function=os.link)
for f in os.listdir(DST):
    if f.endswith(".safetensors"):
        os.remove(os.path.join(DST, f))

st = load_file(os.path.join(SRC, "model.safetensors"))
w = st.pop("lm_head.weight").to(torch.float32)
scale = st.pop("lm_head.weight_scale").to(torch.float32)
st["lm_head.weight"] = (w * scale).to(torch.bfloat16)
print("lm_head ->", tuple(st["lm_head.weight"].shape), st["lm_head.weight"].dtype)

save_file(st, os.path.join(DST, "model.safetensors"), metadata={"format": "pt"})

cfg = json.load(open(os.path.join(DST, "config.json")))
q = cfg.get("quantization_config", {})
ig = q.setdefault("ignore", [])
if not any("lm_head" in x for x in ig):
    ig.append("lm_head")
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)
print("saved + ignore:", q["ignore"])
