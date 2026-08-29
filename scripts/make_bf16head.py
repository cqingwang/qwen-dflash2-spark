import json, os, shutil, torch
from safetensors.torch import load_file, save_file

SRC = os.environ.get("SRC", "/work/unsloth-nvfp4")
DST = os.environ.get("DST", "/work/unsloth-nvfp4-bf16head")

if not os.path.isdir(DST):
    shutil.copytree(SRC, DST, copy_function=os.link)
for f in os.listdir(DST):
    if f.endswith(".safetensors") and f != "model_mtp.safetensors":
        os.remove(os.path.join(DST, f))

mtp_src = os.path.join(SRC, "model_mtp.safetensors")
mtp_dst = os.path.join(DST, "model_mtp.safetensors")
if os.path.exists(mtp_src) and not os.path.exists(mtp_dst):
    os.link(mtp_src, mtp_dst)

st = load_file(os.path.join(SRC, "model.safetensors"))
w = st.pop("lm_head.weight").to(torch.float32)
scale = st.pop("lm_head.weight_scale").to(torch.float32)
st["lm_head.weight"] = (w * scale).to(torch.bfloat16)
print("lm_head ->", tuple(st["lm_head.weight"].shape), st["lm_head.weight"].dtype)

save_file(st, os.path.join(DST, "model.safetensors"), metadata={"format": "pt"})
os.chmod(os.path.join(DST, "model.safetensors"), 0o644)

cfg = json.load(open(os.path.join(DST, "config.json")))
q = cfg.get("quantization_config", {})
groups = q.get("config_groups", {})
group = groups.get("group_0", {})
targets = group.get("targets", [])
group["targets"] = [target for target in targets if "lm_head" not in target]
if group:
    groups["group_0"] = group
    q["config_groups"] = groups
ig = q.setdefault("ignore", [])
if not any("lm_head" in x for x in ig):
    ig.append("lm_head")
config_path = os.path.join(DST, "config.json")
config_tmp = config_path + ".new"
with open(config_tmp, "w") as config_file:
    json.dump(cfg, config_file, indent=2)
os.replace(config_tmp, config_path)
os.chmod(config_path, 0o644)
print("saved + ignore:", q["ignore"])
