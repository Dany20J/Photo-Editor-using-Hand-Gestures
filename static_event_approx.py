
from event import *
import re

class StaticEventApprox:
    def __init__(self, events_interval=10) -> None:
        self.events_acc = []
        self.events_interval = events_interval


    def get_event_approx(self):
        if len(self.events_acc) % self.events_interval == 0:
            event_to_cnt = self.events_counter()
            sorted_events_cnt = sorted(
                list(zip(event_to_cnt.values(), event_to_cnt.keys())), reverse=True)

            most_recurrent_event = sorted_events_cnt[0][1]
            if most_recurrent_event == str(NoEvent()):
                return NoEvent()
            elif most_recurrent_event == str(ClosedHandEvent(None)):
                return ClosedHandEvent(self.get_center_avg())
            else:
                return UpFingersEvent(self.get_center_avg(), int(re.search(r"\d", most_recurrent_event).group(0)))
        else:
            return NoDecisionYetEvent()

    def events_counter(self):
        dict_cnt = {}
        num_events = 0
        for event in reversed(self.events_acc):
            if dict_cnt.get(str(event), -1) == -1:
                dict_cnt[str(event)] = 0
            dict_cnt[str(event)] += 1
            num_events += 1
            if num_events == self.events_interval:
                break
        return dict_cnt

    def add_new_event(self, event):
        self.events_acc.append(event)

    def get_center_avg(self):
        center_sum = np.array([0, 0], np.float32)
        no_events_number = 0

        num_events = 0
        for event in reversed(self.events_acc):
            if str(event) == str(NoEvent()):
                no_events_number += 1
            else:
                center_sum += event.center

            num_events += 1
            if num_events == self.events_interval:
                break
        return center_sum // (num_events - no_events_number)    
    def clear(self):
        self.events_acc = []