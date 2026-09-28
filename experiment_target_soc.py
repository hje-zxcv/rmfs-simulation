import random
import numpy as np
import sys
import io
import simpy
import Layout
from Main import RMFS_Model


def run_single_experiment_quiet(intensity, seed, numCycle, cycleSeconds, numOrderPerCycle=23, enable_target_control=False):
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
        sim.enableTargetSOCControl = enable_target_control
        sim.MultiCycleVRP(numCycle=numCycle, cycleSeconds=cycleSeconds, printOutput=False, numOrderPerCycle=numOrderPerCycle)
    finally:
        sys.stdout = _stdout_backup
    return {
        'intensity': intensity, 'seed': seed,
        'enable_target_control': enable_target_control,
        'total_cost': sim.totalElectricityCost,
        'throughput': sim.totalPodNumber,
        'total_charge_events': sum(r.chargeCount for r in sim.Robots),
    }