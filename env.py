import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.io import savemat
from queue_setup import create_packets, sort_packets, Packet,FCFS_fin, PacketOutput, get_outputs_by_source, calculate_average_age_information, get_age_of_information_updates,calculate_average_successrate
import dataclasses
import numpy as np

from dataclasses import dataclass
from typing import Protocol, Any, NamedTuple,List


def FCFS_Model(mu, p0, p1):
    p = p0 + p1

    def delta(pi):
        return (1 / mu) * (1 + p) * (1 / pi)

    return delta(p0) + delta(p1)


def LCFS_W_Model(mu, p0, p1):
    p = p0 + p1

    def aw(p):
        return ((((1 + p + p ** 2) ** 2) + 2 * (p ** 3)) / ((1 + p + p ** 2) * ((1 + p) ** 2)))

    def delta(pi):
        return (1 / mu) * (aw(p) + (1 + ((p ** 2) / (1 + p))) * (1 / pi))

    return delta(p0) + delta(p1)

POLICIES = [
    ("fifo", FCFS_fin())
]


def fig2(p: float, p0_high_piority: float, p_else: float, time: float,
         service_time:float, device_num:int, high_device_id:list,
         high_queue_length:int, low_queue_length:int, ylim=None,
         plot_original_functions: bool = False):
    service_time_mean = 1

    p0 = p0_high_piority

    data = {
        policy_name: [] for policy_name, _ in POLICIES
    }

    p1 = p_else
    packet_lists_created: List[Packet] = []
    packet_lists_created.append(create_packets(max_arrival_time=time, source=0, arrival_rate=p0, service_time_mean=service_time, seed=0))
    for i in range(1,device_num+1):
        packet_lists_created.append(create_packets(max_arrival_time=time, source=i, arrival_rate=p1, service_time_mean=service_time, seed=i))
    packets = sort_packets(packet_lists_created)

    for policy_name, policy in POLICIES:
        outputs = policy.simulate(packets,high_device_id,high_queue_length,low_queue_length)
        list_aoi=[]
        loss_prob=[]
        list_energy=[]
        for i in range(0,device_num+1):
            a0 = calculate_average_age_information(outputs, source=i)
            if a0==-1:
                a0=0.5*time**2
            s0,e0 = calculate_average_successrate(packets,outputs, source=i)
            list_aoi.append(a0)
            loss_prob.append(s0)
            list_energy.append(e0)

        print(policy_name, sum(list_aoi), sum(loss_prob), sum(list_energy))
        data[policy_name].append((p0, sum(list_aoi)))



    return data,list_aoi,loss_prob,list_energy



# a,b,c,d=fig2(p=1, p0_high_piority=0.05, p_else=1, time=100, service_time=1,device_num=10,high_device_id=[0],high_queue_length=5,low_queue_length=3)
#
# print(a,'\n',b,'\n',c,'\n',d)
#
# a,b,c,d=fig2(p=1, p0_high_piority=0.05, p_else=1, time=100, service_time=1,device_num=10,high_device_id=[0,1,4,5],high_queue_length=5,low_queue_length=3)
#
# print(a,'\n',b,'\n',c,'\n',d)