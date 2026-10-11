# dwell_table.py : dwell_*.json 결과를 표로. 같은 조건의 시드가 여러 개면 SOH 0.75/0.95 비율을 시드별로 구해 평균냄
import glob, json
import numpy as np
import pandas as pd
rows=[]
for f in sorted(glob.glob("dwell_*.json")):
    j=json.load(open(f,encoding="utf-8")); d=j["diag"]; ss=d.get("status_share"); ss=ss if isinstance(ss,dict) else {}
    rows.append({"시간":f.split("_")[1],"시드":j.get("seed",100),"작업량":j["workload"],"SOH":j["soh"],"체류(초)":j["dwell"],"주기(초)":j["cycle"],"모드":j["mode"],"로봇":j["robots"],
                 "처리량/일":round(j["thr"]),"팟방문/일":round(d.get("pod_visits_per_day") or 0),
                 "팟1회(초)":None if d.get("T_sec_per_pod") is None else round(d["T_sec_per_pod"],1),
                 "스테이션이용률":round(d.get("station_util",0),2),
                 "충전비중":ss.get("charging"),"일하는비중":ss.get("extract"),"쉬는비중":ss.get("rest",0),
                 "백로그끝":d.get("backlog_end"),"백로그기울기":None if d.get("backlog_slope_per_hour") is None else round(d["backlog_slope_per_hour"],2)})
df=pd.DataFrame(rows)
print(df.to_string(index=False))
key=["시간","작업량","체류(초)","주기(초)","모드","로봇"]
if len(df):
    out=[]
    for k,g in df.groupby(key):
        p=g.pivot_table(index="시드",columns="SOH",values="처리량/일")
        if 0.95 in p.columns and 0.75 in p.columns:
            r=(p[0.75]/p[0.95]).dropna()
            if len(r):
                out.append(dict(zip(key,k)) | {"시드수":len(r),"비율(평균)":round(r.mean(),4),
                    "비율(최소~최대)":f"{r.min():.3f}~{r.max():.3f}",
                    "표준오차":None if len(r)<2 else round(r.std(ddof=1)/np.sqrt(len(r)),4)})
    if out:
        print("\n[핵심] SOH 0.75 / SOH 0.95 처리량 비율 (같은 시드끼리 짝지어 계산, 0.95 미만이면 노화 영향이 뚜렷)")
        print(pd.DataFrame(out).to_string(index=False))
