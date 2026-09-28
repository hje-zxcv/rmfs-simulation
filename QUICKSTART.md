# Quick Start (For Team Members)

## 1. Setup (run in Anaconda Prompt)

```
cd C:\Users\본인계정
git clone https://github.com/hje-zxcv/rmfs-simulation.git
cd rmfs-simulation
pip install -r requirements.txt
```

## 2. Run simulation via GUI (run in Jupyter Notebook)

```python
%cd C:\Users\본인계정\rmfs-simulation
!python -u Interface.py
```

## 3. Run TOU experiment with repeats (run in Jupyter Notebook)

```python
import sys, io, random
import numpy as np
import pandas as pd
import simpy
from Main import RMFS_Model
import Layout

def run_single_experiment_quiet(intensity, seed, numCycle, cycleSeconds, numOrderPerCycle=23):
    random.seed(seed)
    np.random.seed(seed)
    _stdout_backup = sys.stdout
    sys.stdout = io.StringIO()
    try:
        env = simpy.Environment()
        rows, columns = 19, 61
        network, pos = Layout.create_rectangular_network_with_attributes(columns, rows)
        Layout.place_shelves_automatically(network, shelf_dimensions=(4, 2), spacing=(1, 1))
        sim = RMFS_Model(env=env, network=network, TaskAssignmentPolicy="vrp", ChargePolicy="pearl")
        sim.createPods()
        sim.createSKUs()
        sim.createChargingStations([(0, 12)])
        sim.createOutputStations([(20, 12), (40, 6)])
        sim.fillPods()
        sim.distanceMatrixCalculate()
        sim.createRobots([(0, 0), (40, 0)])
        sim.touIntensity = intensity
        sim.MultiCycleVRP(numCycle=numCycle, cycleSeconds=cycleSeconds, printOutput=False, numOrderPerCycle=numOrderPerCycle)
    finally:
        sys.stdout = _stdout_backup
    return {
        'intensity': intensity, 'seed': seed,
        'total_cost': sim.totalElectricityCost,
        'throughput': sim.totalPodNumber,
        'total_charge_events': sum(r.chargeCount for r in sim.Robots),
    }

cycleSeconds = 900
numCycle = (24 * 3600 // cycleSeconds) * 2
intensities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_REPEATS = 10  # 반복 횟수

results = []
count = 0
total = len(intensities) * N_REPEATS
for intensity in intensities:
    for seed in range(N_REPEATS):
        count += 1
        r = run_single_experiment_quiet(intensity, seed, numCycle, cycleSeconds)
        results.append(r)
        print(f"[{count}/{total}] intensity={intensity}, seed={seed} -> "
              f"cost={r['total_cost']:.1f}, throughput={r['throughput']}, "
              f"charge_events={r['total_charge_events']}")

df_results = pd.DataFrame(results)
df_results.to_csv('tou_experiment_results.csv', index=False)
print("저장 완료: tou_experiment_results.csv")
```

## 4. 목표 충전량 통제 실험 (신규)

`git pull`로 최신 코드를 받은 후, Jupyter에서:

\`\`\`python
%cd "본인 경로\rmfs-simulation-main"
from experiment_target_soc import run_single_experiment_quiet
import pandas as pd

results = []
NUM_CYCLE = 192
CYCLE_SECONDS = 900
INTENSITY = 본인이 맡은 값  # 0.8 또는 1.0

for seed in range(본인이 맡은 seed 시작, 본인이 맡은 seed 끝):
    for enable in [False, True]:
        r = run_single_experiment_quiet(
            intensity=INTENSITY, seed=seed,
            numCycle=NUM_CYCLE, cycleSeconds=CYCLE_SECONDS,
            enable_target_control=enable
        )
        results.append(r)
        print(r)

df = pd.DataFrame(results)
df.to_csv(f"target_soc_results_intensity_{INTENSITY}.csv", index=False)
\`\`\`


