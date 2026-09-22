# RMFS Simulation

This is a Robotic Mobile Fulfillment System (RMFS) simulation written to try different strategies for assignment and routing problems in a robotic warehouse. The project is a part of [IE497](https://catalog.metu.edu.tr/course.php?course_code=5680497) - [IE498](https://catalog.metu.edu.tr/course.php?course_code=5680498) Systems Design course of [METU Industrial Engineering](https://ie.metu.edu.tr) department.

The simulation is working with the described features below. You can run the simulation using the guide below. For further features, you can open an issue, pull request or contact me.

**Language:** Python

**Libraries:** pandas, numpy, simpy, networkx, gurobi, google or tools, pydoe3, tkinter

# Step by Step Guide

Run the `interface.py` file.

<div align="center">
<img src="Interface.png" width="600" alt="Interface">
</div>


**Warehouse & Simulation Settings:**

1. Horizontal & Vertical Ailes:

Number of inner ailes. Shelves/pods are created as a group of 2x4=8.

e.g. When there are 2 horizontal and 2 vertical ailes, there are 4 corridors in each direction and a layout of 10x16 is created.

<div align="center">
<img src="Warehouse.png" width="450" alt="Warehouse Layout">
</div>

2. Station Amounts and Locations:

Pick stations and charge stations are created. Station amount of one type cannot exceed the number of ailes in that direction.
Total station to be placed on a side cannot exceed the number of ailes intersecting with that side.

3. Cycle Amount and Runtime:

The tasks and routes are determined in cycles. Cycle runtime is in seconds, so if a cycle runtime is determined as 900 seconds and the number of cycles is 4, total simulation time is an hour.

**Robot Settings:**

All robots in the simulation are of the same type.

1. Robot Amount: Number of robots in the simulation.

2. Charging Rate (Ah): Charging rate of the robots' battery.

3. Maximum Battery (Ah): Maximum charge capacity of the robots' battery.

**Charge Policy Settings:**

1. Pearl Rate: If the charge difference between charge-seeking and charging robot is above the pearl rate (replacement threshold), charging robot gives its place in charging station to the charge-seeker.

2. Rest Rate: The percentage charge level of the robot to leave the charge station.

3. Charge Flag Rate: The percentage charge level of the robot to seek for an empty charge station.

4. Max Charge Rate: The percentage charge level of the robot to leave the charge station.

Charge flow is as follows:
<div align="center">
<img src="ChargeFlow.png" width="600" alt="Charge Flow">
</div>

**Taguchi Experiment Frame:**

1. Enable Taguchi Experiment: In order to conduct experimentation, in other words, run the simulation once and get different configured simulation results, you need to check this box.

2. Pick Station, Charge Station Amount: In order to try different amounts of stations, check the boxes and enter the second amount.

3. Charge Flag Rate, Max Charge Rate, Pearl Rate: You can input secondary values for these features by checking their boxes.

4. Experiment Objective: Currently does not affect the output. Regardless of your choice, experiments result for all metrics.

If you're all set, you can click "Run Simulation" and you'll receive an output Excel.
Current experiments file are filled with experiments from our senior year, you can clear those files before starting experimentations.

When you choose what you want to experiment, an experiment matrix is created using [PyDOE3](https://pydoe3.readthedocs.io/en/latest/). It is a 2-Level Full-Factorial experiment. You can see the experiment matrix in the output Excel.

**Experiment Outputs:**

For each robot, time-dependent:
- Task
- Distance traveled
- Charge level
- Number of completed tasks
- Number of remaining tasks

Additionally, throughout the simulation:
- Number of products collected by each station
- Number of times each robot went to charge
- Number of times each robot moved to charge

---

# Update Notes (2026-09)

This fork includes the following fixes and additions on top of the original repository:

## Bug Fixes
- Fixed filename case mismatch (`main` → `Main`) causing import errors on case-sensitive systems.
- Removed leftover markdown code fences (` ``` `) accidentally left in `Main.py` and `config.py`.
- Fixed missing `_STR` suffix inconsistencies in `config.py` variable names referenced by `Interface.py`.
- Fixed indentation error in `Interface.py`'s Taguchi experiment loop.
- Restored core methods in `Main.py` (`updateCharge`, `collectTimeStat`, `calculateObservationStat`, `orderGenerator`, `distanceMatrixCalculate`, etc.) that were missing due to a prior refactor. The full original implementation was recovered from an earlier commit in the repository history.
- Fixed tuple unpacking bug in `podSelectionHungarian` and function name mismatch (`MultiCycleVRP` vs `TaguchiVRP`).

## Setup Notes
- Install dependencies with `pip install -r requirements.txt`.
- If you hit a NumPy/pandas import error, make sure NumPy is pinned below version 2 (`pip install "numpy<2" --force-reinstall`), then **restart your kernel/terminal**.
- `xlsxwriter` is required for Excel output and was not listed in the original README.

## New Feature: Time-of-Use (TOU) Aware Charging Scheduling
Added a set of functions in `Main.py` to make robot charging schedules responsive to time-of-use electricity pricing:

- `get_current_hour(env_now)`: converts simulation time (seconds) into a 0–23 hour value.
- `get_electricity_price(hour)`: returns the electricity price (KRW/kWh) based on Korea's 2026 industrial (Eul) tariff (High-voltage A, Option 3, summer season), split into off-peak / mid-peak / peak hours.
- `get_dynamic_charge_flag_rate(base_rate, hour, rest_rate, intensity)`: adjusts each robot's charge-seeking threshold by time of day and a tunable `intensity` parameter (0.0 = baseline, 1.0 = max effect), while always keeping a safety margin above the robot's mandatory rest threshold.
- `RMFS_Model.get_effective_charge_flag_rate(robot)`: helper used by `Entities.py` to fetch the dynamically adjusted threshold at charge-decision time.
- `RMFS_Model.totalElectricityCost`: accumulates electricity cost (KRW) over a simulation run based on each charging event's timing and the price at that hour.

This enables comparing a baseline (fixed threshold, `intensity=0`) against TOU-aware policies (`intensity>0`) to study the cost-vs-throughput trade-off of shifting charging load toward cheaper night-time hours.
