import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.io import savemat
from queue_setup import create_packets, sort_packets, Packet,FCFS_fin, PacketOutput, get_outputs_by_source, calculate_average_age_information, get_age_of_information_updates



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
    # ("fifo", FCFS()),
    ("fifo", FCFS_fin())
]


def fig2(p: float, p0_min: float, p0_max: float, p0_step: float, time: float, service_time:float,device_num:int,high_device_id:list,high_queue_length:int,low_queue_length:int, ylim=None,plot_original_functions: bool = False):
    service_time_mean = 1

    p0 = p0_min

    data = {
        policy_name: [] for policy_name, _ in POLICIES
    }

    while p0 <= p0_max:
        p1 = (p - p0)/device_num

        packets = sort_packets(
            create_packets(max_arrival_time=time, source=0, arrival_rate=p0, service_time_mean=service_time, seed=0),
            create_packets(max_arrival_time=time, source=1, arrival_rate=p1, service_time_mean=service_time, seed=1),
            create_packets(max_arrival_time=time, source=2, arrival_rate=p1, service_time_mean=service_time, seed=2),
            create_packets(max_arrival_time=time, source=3, arrival_rate=p1, service_time_mean=service_time, seed=3),
            create_packets(max_arrival_time=time, source=4, arrival_rate=p1, service_time_mean=service_time, seed=4),
            create_packets(max_arrival_time=time, source=5, arrival_rate=p1, service_time_mean=service_time, seed=5),
            create_packets(max_arrival_time=time, source=6, arrival_rate=p1, service_time_mean=service_time, seed=6),
            create_packets(max_arrival_time=time, source=7, arrival_rate=p1, service_time_mean=service_time, seed=7),
            create_packets(max_arrival_time=time, source=8, arrival_rate=p1, service_time_mean=service_time, seed=8),
            create_packets(max_arrival_time=time, source=9, arrival_rate=p1, service_time_mean=service_time, seed=9),
            create_packets(max_arrival_time=time, source=10, arrival_rate=p1, service_time_mean=service_time, seed=10),
        )

        for policy_name, policy in POLICIES:
            outputs = policy.simulate(packets,high_device_id,high_queue_length,low_queue_length)
            sum_aoi=0
            for i in range(0,device_num+1):
                print(p0,i, len(outputs))
                a0 = calculate_average_age_information(outputs, source=i)
                sum_aoi+=a0
            print(policy_name, sum_aoi)
            data[policy_name].append((p0, sum_aoi))

        p0 += (p0_max-p0_min)/p0_step

    for policy_name, points in data.items():
        file_name = str(service_time)+'data.mat'
        savemat(file_name, {'points':points,str(service_time)+'x':[x for x, _ in points],str(service_time)+'y':[y for _, y in points]})
        plt.plot([x for x, _ in points], [y for _, y in points], label=f'{policy_name}', marker='o', markersize=3)


    plt.title(f'p={p}')

    plt.xlabel('p0')
    plt.ylim(ylim)
    plt.ylabel('Sum Average AoI')
    plt.legend()
    plt.show()

    return data


high_device_id=[0,1]
high_queue_length=5
low_queue_length=5

data1 = fig2(p=1, p0_min=0.1, p0_max=0.9, p0_step=8, time=25000,service_time=1, device_num=10,high_device_id=high_device_id,high_queue_length=high_queue_length,low_queue_length=low_queue_length,ylim=(0, 200),plot_original_functions=True)
data1 = fig2(p=1, p0_min=0.1, p0_max=0.9, p0_step=8, time=25000,service_time=0.6, device_num=10,high_device_id=high_device_id,high_queue_length=high_queue_length,low_queue_length=low_queue_length,ylim=(0, 200),plot_original_functions=True)
data1 = fig2(p=1, p0_min=0.1, p0_max=0.9, p0_step=8, time=25000,service_time=0.3, device_num=10,high_device_id=high_device_id,high_queue_length=high_queue_length,low_queue_length=low_queue_length,ylim=(0, 200),plot_original_functions=True)

# for policy_name, points in data1.items():
#     plt.plot([x for x, _ in points], [y for _, y in points], label=f'{policy_name}', marker='o', markersize=3)
# for policy_name, points in data2.items():
#     plt.plot([x for x, _ in points], [y for _, y in points], label=f'{policy_name}', marker='o', markersize=3)
# for policy_name, points in data3.items():
#     plt.plot([x for x, _ in points], [y for _, y in points], label=f'{policy_name}', marker='o', markersize=3)
#
# plt.title(f'p={p}')
#
# plt.xlabel('p0')
# plt.ylim((0, 100))
# plt.ylabel('Sum Average AoI')
# plt.legend()
# plt.show()
