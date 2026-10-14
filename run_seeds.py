# run_seeds.py : 처리량 상한(PICK_ITEM_CAP=16) 측정을 시드 범위로 돌림. item_run.py와 같은 폴더(Main.py 있는 곳)에서 실행.
# 사용법: python run_seeds.py <시작시드> <끝시드(포함)> [동시실행수=2]
#   지은: python run_seeds.py 100 109 2
#   효빈: python run_seeds.py 110 119 2
#   서현: python run_seeds.py 120 129 2
# 이미 끝난 결과(json)는 건너뛰므로, 중간에 꺼져도 같은 명령을 다시 실행하면 이어서 진행됨.
import sys, os, subprocess, glob
from concurrent.futures import ThreadPoolExecutor
if len(sys.argv) not in (3, 4): sys.exit(__doc__)
S0, S1 = int(sys.argv[1]), int(sys.argv[2]); NW = int(sys.argv[3]) if len(sys.argv) == 4 else 2
W, HOURS, BASE, PER, CYCLE, MODE, NR, CAP = "4.0", "6", "2", "0.36", "300", "B", "4", "16"
SOHS = ["0.95", "0.85", "0.80", "0.75"]
def done(seed, soh):
    sfx = "" if seed == 100 else f"_seed{seed}"
    return os.path.exists(f"item_{HOURS}h_w{float(W)}_s{float(soh)}_b{BASE}_p{PER}_c{CYCLE}_{MODE}_r{NR}_cap{CAP}{sfx}.json")
def job(a):
    seed, soh = a
    if done(seed, soh): return f"skip seed{seed} SOH{soh}"
    env = dict(os.environ, PICK_ITEM_CAP=CAP)
    r = subprocess.run([sys.executable, "item_run.py", W, soh, HOURS, BASE, PER, CYCLE, MODE, NR, str(seed)],
                       env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return f"seed{seed} SOH{soh}: " + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"실패 rc={r.returncode}")
jobs = [(s, h) for s in range(S0, S1 + 1) for h in SOHS]
print(f"총 {len(jobs)}건, 동시 {NW}개")
with ThreadPoolExecutor(NW) as ex:
    for msg in ex.map(job, jobs): print(msg, flush=True)
print("끝")
