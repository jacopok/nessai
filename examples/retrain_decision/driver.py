import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
cfgs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
def go(c):
    p = subprocess.run([sys.executable, "harness.py", json.dumps(c)], capture_output=True, text=True)
    out = p.stdout.strip().splitlines()
    return out[-1] if out and p.returncode == 0 else f"FAIL {c['name']}: {p.stderr[-2000:]}"
with ThreadPoolExecutor(nproc) as ex:
    for r in ex.map(go, cfgs):
        print(r, flush=True)
