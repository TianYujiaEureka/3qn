import dataclasses
import numpy as np

from dataclasses import dataclass
from typing import Protocol, Any, NamedTuple,List


class Stack:
    def __init__(self):
        # the last item is the top of the stack
        self.items: List[Any] = []

    def push(self, item: Any):
        self.items.append(item)

    def pop(self) -> Any:
        return self.items.pop(-1)

    def empty(self) -> bool:
        """returns true when the stack is empty"""
        return len(self.items) == 0


class Queue:
    def __init__(self,length_max=1000):
        # the last item is the top of the stack
        self.items: List[Any] = []
        self.length_max = length_max;

    def get_length(self) -> int:
        return len(self.items)

    def set_length(self,length_max:int):
        if length_max >=self.length_max:
            self.length_max = length_max;
        else:
            self.items=self.items[:length_max];
            self.length_max = length_max;
    def push(self, item: Any):
        if (self.items.__len__() < self.length_max):
            self.items.append(item)

    def insert(self, item: Any):
        # insert at the front of the queue
        if(self.items.__len__()<self.length_max):
            self.items.insert(0, item)


    def pop(self) -> Any:
        if not self.empty():
            return self.items.pop(0)
        return None

    def empty(self) -> bool:
        """returns true when the queue is empty"""
        return len(self.items) == 0


@dataclass
class Packet:
    arrival_time: float
    service_time: float
    source: int


@dataclass
class PacketOutput:
    source: int
    arrival_time: float
    service_end_time: float


class Simulation(Protocol):
    def simulate(self, packets:  List[Packet]) -> List[PacketOutput]:
        pass


class FCFS_fin:
    def simulate(self, packets: List[Packet],high_device_id:list,high_queue_length:int,low_queue_length:int) -> List[PacketOutput]:
        # check there's at least one packet to simulate
        if len(packets) == 0:
            return []

        # copy the packets since this method will modify the service_time
        # as the packets are processed.
        packets = [dataclasses.replace(packet) for packet in packets]

        # run the simulation
        last_update: List[float] = [-1, -1]
        queue_high = Queue(high_queue_length+low_queue_length)
        queue_low = Queue(low_queue_length)
        queue_length_high_now = 0;
        queue_length_low_now =0;

        output: List[PacketOutput] = []
        clock=0
        for packet in packets:
            if packet.arrival_time<=clock :
                queue_length_high_now = queue_high.get_length()
                queue_length_low_now = low_queue_length - max(queue_length_high_now - high_queue_length, 0);
                queue_low.set_length(queue_length_low_now)
                #没传完的时间里到的数据包都进队
                if (packet.source in high_device_id):
                    queue_high.push(packet)
                else:
                    queue_low.push(packet)

            else:
                if not queue_high.empty():
                    last_packet = queue_high.pop()
                elif not queue_low.empty():
                    last_packet = queue_low.pop()
                else:
                    last_packet=packet
                clock=max(last_packet.arrival_time,clock) + last_packet.service_time
                output.append(
                    PacketOutput(
                        source=last_packet.source,
                        arrival_time=last_packet.arrival_time,
                        service_end_time=clock,
                    )
                )
        return output




def sort_packets(packet_lists: List[Packet]) -> List[Packet]:
    """sort multiple lits of packets by ascending arrival time"""
    packets = []
    for packet_list in packet_lists:
        packets.extend(packet_list)
    return sorted(packets, key=lambda packet: packet.arrival_time)


def create_packets(
    max_arrival_time: float,
    source: int,
    arrival_rate: float,
    service_time_mean: float,
    seed=None,
) -> List[Packet]:
    """create packets for a source with an arrival rate, and service time mean"""
    packets: List[Packet] = []
    arrival_time: int = 0
    # np.random.seed(seed)
    rng: np.random.Generator = np.random.default_rng(seed)

    while True:
        arrival_time += rng.exponential(scale=1.0 / arrival_rate)
        if arrival_time > max_arrival_time:
            break

        packets.append(
            Packet(
                arrival_time=arrival_time,
                service_time=service_time_mean,
                source=source,
            )
        )

    return packets


def get_outputs_by_source(
    packets: List[PacketOutput], source: int
) -> List[PacketOutput]:
    return [p for p in packets if p.source == source]


class AoiUpdate(NamedTuple):
    time: float  # when this age of information was valid
    age: float  # the age of information

def calculate_average_successrate(
    input:List[Packet], outputs: List[PacketOutput], source: int
) -> float:
    all_num=get_outputs_by_source(input,source)
    success_num=get_outputs_by_source(outputs,source)
    return len(success_num)/len(all_num),len(success_num)


def calculate_average_age_information(
    outputs: List[PacketOutput], source: int
) -> float:
    updates = get_age_of_information_updates(outputs, source)
    if(len(updates)<=2):
       return -1
    total = 0
    total_delta_time = 0

    for i in range(1, len(updates) - 1, 2):
        prev_update = updates[i - 1]
        update = updates[i]

        delta_time = update.time - prev_update.time
        total += ((update.age + prev_update.age) / 2) * delta_time
        total_delta_time += delta_time
    if total_delta_time==0:
        print(total_delta_time)
    return total /total_delta_time


def get_age_of_information_updates(
    outputs_in: List[PacketOutput], source: int
) -> List[AoiUpdate]:
    """helper method to get "aoi updates" which are used to calculate average aoi
    and plot aoi over time"""

    updates = [AoiUpdate(time=0, age=0)]  # simulation time, age of information
    outputs_out = get_outputs_by_source(outputs_in, source)

    prev = PacketOutput(arrival_time=0, service_end_time=0, source=-1)  # dummy packet
    for packet in outputs_out:
        time = packet.service_end_time  # when the age of information changes
        updates.append(
            AoiUpdate(
                time=time,
                age=prev.service_end_time
                - prev.arrival_time
                + (packet.service_end_time - prev.service_end_time),
            )
        )
        updates.append(
            AoiUpdate(time=time, age=packet.service_end_time - packet.arrival_time)
        )
        prev = packet

    return updates
