# screen_run.py : 포화 점검 1칸(작업량 1개 x SOH 1개)을 터미널에서 실행. 여러 창에서 동시에 돌릴 수 있음.
# 사용법: python screen_run.py <작업량> <SOH> [시간=12]     예) python screen_run.py 5.0 0.95
# dwell_run.py : 체류시간/주문주기/로봇대수를 바꿔 포화 여부를 1칸씩 점검 (터미널 실행)
# item_run.py : 체류 = 기본초 + 개당초 x (이번 방문 처리 건수). 나머지는 dwell_run.py와 같음.
# 사용법: python item_run.py <작업량> <SOH> <시간> <기본초> <개당초> <주기초> <모드A|B> <로봇수> [시드=100]
#   예) python item_run.py 4.0 0.95 12 2 0.36 300 B 4      (평균 11건일 때 약 6초)
#       python item_run.py 4.0 0.95 12 3 0.64 300 B 4      (평균 11건일 때 약 10초)
#   예) python dwell_run.py 4.0 0.95 12 6 900 A 4      (체류 6초만 추가, 주기 그대로)
#       python dwell_run.py 4.0 0.95 12 6 300 A 4      (주기 300초, 주기당 주문 수 유지 = 시간당 주문 3배)
#       python dwell_run.py 4.0 0.95 12 6 300 B 4      (주기 300초, 하루 총량 유지)
#       python dwell_run.py 4.0 0.95 12 6 900 A 3      (로봇 3대)
import sys
if len(sys.argv) not in (9, 10):
    sys.exit(__doc__)
W, S, HOURS = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
BASE, PER = float(sys.argv[4]), float(sys.argv[5])
CYCLE, MODE, NR = int(sys.argv[6]), sys.argv[7].upper(), int(sys.argv[8])
DWELL = BASE
HOURS = int(HOURS) if HOURS == int(HOURS) else HOURS
SEED = int(sys.argv[9]) if len(sys.argv) == 10 else 100
import os as _os
CAP = int(_os.environ["PICK_ITEM_CAP"]) if _os.environ.get("PICK_ITEM_CAP") else None   # 방문당 처리 상한(건). 환경변수로 지정
OUT = f"item_{HOURS}h_w{W}_s{S}_b{BASE:g}_p{PER:g}_c{CYCLE}_{MODE}_r{NR}" + ("" if CAP is None else f"_cap{CAP}") + ("" if SEED == 100 else f"_seed{SEED}") + ".json"
import os, json, time, random, io, contextlib
import numpy as np
import pandas as pd
import simpy
import Layout
import config
config.PICK_DWELL_BASE = BASE         # Entities.py가 이 값을 읽음
config.PICK_DWELL_PER_ITEM = PER
config.PICK_DWELL_SECONDS = BASE
config.PICK_ITEM_CAP = CAP            # Main.py가 이 값을 읽음 (None이면 상한 없음)
if os.path.exists(OUT):
    sys.exit(f"이미 끝남: {OUT}")
from Main import RMFS_Model, get_current_hour, get_dynamic_charge_flag_rate
from Entities import Robot

SOH0, ALPHA = 0.9564, 0.0000688
RATED = 60.0                       # 새 사양: 48V x 30Ah = 1,440Wh = 24V x 60Ah (24V 환산)
N_ROBOTS = NR
BASE_ORDERS = 30
INIT_SOC_RANGE = (0.30, 0.75)

def _diagnostics(df, obs, days, qsamples, caps, flag, rest_rate, intensity):
    """지표 계산 (일부가 실패해도 본 결과는 유지되도록 항목별로 보호)"""
    d = {}
    try:
        d["status_share"] = df["robotStatus"].value_counts(normalize=True).round(4).to_dict()
    except Exception as e:
        d["status_share"] = f"실패: {e}"
    try:
        d["charge_count_per_day"] = float(obs[obs["Statistics"].str.contains("NumberOfCharge")]["Value"].sum()) / days
    except Exception as e:
        d["charge_count_per_day"] = None
    try:
        last, ok = 0.0, True
        for rid, g in df.groupby("robotID"):
            s = g.sort_values("time")["stepsTaken"].to_numpy(dtype=float)
            if np.any(np.diff(s) < 0): ok = False
            last += s[-1]
        d["steps_per_day"] = (last / days) if ok else None      # stepsTaken이 누적값일 때만 의미 있음
    except Exception as e:
        d["steps_per_day"] = None
    try:
        t = df.groupby("time")["remainingTasks"].first().astype(float)
        t = t.dropna()
        hours = t.index.to_numpy(dtype=float) / 3600
        vals = t.to_numpy()
        m24 = hours >= hours.max() - 24
        slope = float(np.polyfit(hours[m24], vals[m24], 1)[0]) if m24.sum() > 2 else None
        d["backlog_end"] = float(vals[-1])
        d["backlog_mean_last12h"] = float(vals[hours >= hours.max() - 12].mean())
        d["backlog_slope_per_hour"] = slope                      # 양수로 계속 커지면 포화 신호
    except Exception as e:
        d["backlog_end"] = d["backlog_mean_last12h"] = d["backlog_slope_per_hour"] = None
    try:
        q = np.asarray(qsamples, dtype=float)
        d["queue_mean"] = float(q.mean()); d["queue_max"] = float(q.max())
        d["queue_nonempty_share"] = float((q > 0).mean())
    except Exception as e:
        d["queue_mean"] = d["queue_max"] = d["queue_nonempty_share"] = None
    try:
        capmap = {i: float(c) for i, c in enumerate(caps)}
        cap_arr = df["robotID"].astype(int).map(capmap)
        # 시각별 '실제 적용되는' 충전 기준 (목표 SOC 제어가 심야엔 높이고 저녁 피크엔 낮춤)
        eff = {t: get_dynamic_charge_flag_rate(flag, get_current_hour(t), rest_rate, intensity)
               for t in df["time"].unique()}
        thr_arr = df["time"].map(eff).astype(float) * cap_arr
        low = (df["robotStatus"] == "rest") & (df["batteryLevel"].astype(float) < thr_arr)
        d["lowbat_rest_share"] = float(low.mean())     # 충전 기준 아래인데 쉬고 있는(=충전 차례를 못 받은) 시간 비중
    except Exception as e:
        d["lowbat_rest_share"] = None
    return d

def simulate_block_diag(caps, hours=48, reps=1, n_orders=BASE_ORDERS, seed=0,
                        charge_flag_rate=0.25, max_charge_rate=0.75, pearl_rate=0.4,
                        rest_rate=0.1, cycle_seconds=900, tou_intensity=1.0):
    """simulate_block과 같은 구성 + 진단 지표. 결과: efc/전기요금/처리량 + diag"""
    caps = np.asarray(caps, dtype=float)
    num_cycles = int(hours * 3600 / cycle_seconds)
    days = hours / 24
    efc_l, cost_l, thr_l, diag_l = [], [], [], []
    for rep in range(reps):
        s = int(seed) * 1000 + rep
        random.seed(s); np.random.seed(s)
        rng = np.random.default_rng(s)
        env = simpy.Environment()
        network, pos = Layout.create_rectangular_network_with_attributes(61, 19)
        Layout.place_shelves_automatically(network, shelf_dimensions=(4, 2), spacing=(1, 1))
        model = RMFS_Model(env=env, network=network, TaskAssignmentPolicy="vrp", ChargePolicy="pearl")
        model.enableTargetSOCControl = True
        model.touIntensity = tou_intensity
        model.createPods(); model.createSKUs(); model.fillPods()
        model.createOutputStations([(30, 0), (30, 18)])
        model.createChargingStations([(0, 9)])
        start_locations = [(0, 0), (40, 0), (0, 18), (60, 18)][:N_ROBOTS]
        model.Robots = []
        model.ChargeQueue = []
        for idx, (loc, cap) in enumerate(zip(start_locations, caps)):
            robot = Robot(model.env, network_corridors=model.corridorSubgraph, network=model.network,
                          robotID=idx, currentNode=loc, taskList=[],
                          batteryLevel=float(cap), chargingRate=float(cap), Model=model,
                          ChargeFlagRate=charge_flag_rate, MaxChargeRate=max_charge_rate,
                          PearlRate=pearl_rate, RestRate=rest_rate,
                          chargingStationList=model.ChargingStations)
            robot.batteryLevel = float(cap * rng.uniform(*INIT_SOC_RANGE))
            model.Robots.append(robot)
        model.distanceMatrixCalculate()

        qsamples = []                       # 충전 대기열 길이를 1분마다 기록 (읽기만 함)
        _tw = time.time()
        def _sampler(env_, m_):
            k = 0
            while True:
                qsamples.append(len(m_.ChargeQueue))
                k += 1
                if k % 60 == 0:                     # 시뮬 1시간마다 진행 상황을 화면(stderr)에 한 줄 출력
                    _v = sum(getattr(r_, "pod_visits", 0) for r_ in m_.Robots)
                    print(f"[진행] 시뮬 {k//60}/{HOURS}시간 | 실제 경과 {(time.time()-_tw)/60:.1f}분 | 팟방문 누적 {_v}",
                          file=sys.stderr, flush=True)
                yield env_.timeout(60)
        env.process(_sampler(env, model))

        with contextlib.redirect_stdout(io.StringIO()):
            model.MultiCycleVRP(numCycle=num_cycles, cycleSeconds=cycle_seconds,
                                printOutput=False, numOrderPerCycle=n_orders)
        df = model.timeStatDF.sort_values(["robotID", "time"])
        efc = np.zeros(N_ROBOTS); maxrel = np.zeros(N_ROBOTS)
        for rid, g in df.groupby("robotID"):
            b = g["batteryLevel"].to_numpy(dtype=float)
            drops = -np.diff(b); drops[drops < 0] = 0
            efc[int(rid)]    = drops.sum() / RATED
            maxrel[int(rid)] = b.max() / caps[int(rid)]
        if maxrel.max() > 0.93:
            raise RuntimeError(f"로봇 용량 설정이 반영되지 않은 것 같습니다 (max SOC/cap={maxrel.round(3)})")
        pod_visits = sum(getattr(r_, "pod_visits", 0) for r_ in model.Robots)
        pod_items = sum(getattr(r_, "pod_items", 0) for r_ in model.Robots)
        obs = model.calculateObservationStat()
        thr = obs[obs["Statistics"].str.contains("TotalCollect")]["Value"].sum()
        efc_l.append(efc / days); cost_l.append(model.totalElectricityCost / days); thr_l.append(thr / days)
        dg = _diagnostics(df, obs, days, qsamples, caps, charge_flag_rate, rest_rate, tou_intensity)
        dg["pod_visits_per_day"] = pod_visits / days
        dg["pod_items_per_day"] = pod_items / days
        dg["count_mismatch"] = getattr(model, "_cnt_mismatch", 0)   # 처리량 집계의 작업/선택 개수 불일치 횟수(0이어야 정상)
        diag_l.append(dg)
        del model, env, network
        import gc; gc.collect()
    return {"efc_per_day": np.mean(efc_l, axis=0), "cost_per_day": float(np.mean(cost_l)),
            "throughput_per_day": float(np.mean(thr_l)), "diag": diag_l[0] if reps == 1 else diag_l}



import time
t0 = time.time()
n_orders = BASE_ORDERS * W * (CYCLE / 900 if MODE == "B" else 1)
r = simulate_block_diag([S * RATED] * N_ROBOTS, hours=HOURS, reps=1, n_orders=n_orders, seed=SEED, cycle_seconds=CYCLE)
d = r["diag"]
N = d.get("pod_visits_per_day") or 0
ext = (d.get("status_share") or {}).get("extract", 0) if isinstance(d.get("status_share"), dict) else 0
rest = (d.get("status_share") or {}).get("rest", 0) if isinstance(d.get("status_share"), dict) else 0
T = (ext * N_ROBOTS * 86400 / N) if N else None            # 팟 1회당 로봇이 쓰는 시간(초, 체류 포함)
ITEMS = d.get("pod_items_per_day") or 0
IPV = (ITEMS / N) if N else None                            # 방문당 평균 처리 건수
mean_dwell = BASE + PER * (IPV or 0)                         # 평균 체류(초)
station_util = N * mean_dwell / (2 * 86400)                       # 스테이션 2개 기준 이용률(대기열 없는 가정의 부하 지표)
d["T_sec_per_pod"] = T; d["station_util"] = station_util
d["items_per_visit"] = IPV; d["mean_dwell_sec"] = mean_dwell
json.dump({"seed": SEED, "workload": W, "soh": S, "dwell": DWELL, "base": BASE, "per_item": PER, "cap": CAP, "cycle": CYCLE, "mode": MODE, "robots": N_ROBOTS,
           "thr": r["throughput_per_day"], "cost": r["cost_per_day"], "efc": r["efc_per_day"].tolist(), "diag": d},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
print(f"x{W} SOH {S} 체류{BASE:g}+{PER:g}/건 상한{CAP} 주기{CYCLE}s {MODE} 로봇{N_ROBOTS} | 처리량 {r['throughput_per_day']:.0f}/일 | "
      f"팟방문 {N:.0f}/일 | 방문당건수 {IPV if IPV is None else round(IPV,1)} | 평균체류 {round(mean_dwell,1)}초 | 집계건수/처리량 {round(ITEMS/r['throughput_per_day'],2) if r['throughput_per_day'] else None} | 팟1회 {T if T is None else round(T,1)}초 | 스테이션이용률 {station_util:.2f} | "
      f"쉬는비중 {rest} | 백로그기울기 {d.get('backlog_slope_per_hour')} | {time.time()-t0:.0f}초 -> {OUT}")
