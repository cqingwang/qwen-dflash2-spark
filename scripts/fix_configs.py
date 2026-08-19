import json, os, shutil

def rewrite(path, fix):
    # break any hardlink first: write to temp then replace
    tmp = path + ".new"
    cfg = json.load(open(path))
    fix(cfg)
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, path)
    print("fixed", path)

def fix_bf16head(cfg):
    q = cfg["quantization_config"]
    t = q["config_groups"]["group_0"]["targets"]
    q["config_groups"]["group_0"]["targets"] = [x for x in t if "lm_head" not in x]
    ig = q.setdefault("ignore", [])
    if "lm_head" not in ig:
        ig.append("lm_head")

def fix_original(cfg):
    q = cfg["quantization_config"]
    q["ignore"] = [x for x in q.get("ignore", []) if x != "lm_head"]

rewrite("/p/unsloth-nvfp4-bf16head/config.json", fix_bf16head)
rewrite("/p/unsloth-nvfp4/config.json", fix_original)

# show proof
for p in ["/p/unsloth-nvfp4-bf16head/config.json", "/p/unsloth-nvfp4/config.json"]:
    q = json.load(open(p))["quantization_config"]
    print(p.split("/")[2], "| group_0 lm_head target:", any("lm_head" in x for x in q["config_groups"]["group_0"]["targets"]), "| ignore lm_head:", "lm_head" in q.get("ignore", []))
