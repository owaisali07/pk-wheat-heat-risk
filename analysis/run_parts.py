"""Development helper: execute notebook parts in one namespace (python run_parts.py part1 part2 ...)."""
import sys
import time

ns = {"__name__": "__main__"}
for p in sys.argv[1:]:
    t0 = time.time()
    print(f"\n######## {p}", flush=True)
    exec(compile(open(p, encoding="utf-8").read(), p, "exec"), ns)
    print(f"######## {p} done in {time.time() - t0:.0f} s", flush=True)
