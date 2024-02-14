

from abc import ABC, abstractmethod
from event import ClosedHandEvent, UpFingersEvent, LeftToRightLineMotionEvent, RightToLeftLineMotionEvent, TopToBottomLineMotionEvent, BottomToTopLineMotionEvent, CounterClockwiseCircularMotionEvent, ClockwiseCircularMotionEvent, NoEvent
import system_state as ss
import numpy as np

def release(static):
    return type(static) is UpFingersEvent

def exit(static):
    return type(static) is NoEvent
class State(ABC):
    def __init__(self, system_state) -> None:
        super().__init__()
        self.system_state = system_state
        # self.system_state = ss.SystemState()

    @abstractmethod
    def apply_event(self, event):
        pass

    @abstractmethod
    def change_state(self, event):
        pass

    @abstractmethod
    def notify_as_set(self, event):
        pass

    def get_motion_static_events(self, event):
        return event[0], event[1]


class StartingState(State):
    def __init__(self, system_state) -> None:
        super().__init__(system_state)

    def apply_event(self, event):
        static, motion = self.get_motion_static_events(event)

        if type(static) is ClosedHandEvent:      
            self.system_state.selection_callback(1)
        elif (type(static) is UpFingersEvent and static.num_fingers <= 1):
            self.system_state.selection_callback(4)
        elif type(static) is UpFingersEvent and static.num_fingers <= 3:
            self.system_state.selection_callback(2)
        elif type(static) is UpFingersEvent and static.num_fingers  <= 4:
            self.system_state.selection_callback(3)
        elif type(static) is UpFingersEvent and static.num_fingers <= 5:
            self.system_state.selection_callback(6)

    def change_state(self, event):
        static, motion = self.get_motion_static_events(event)

        if (type(static) is UpFingersEvent and static.num_fingers <= 1) or type(static) is ClosedHandEvent:      
            self.system_state.set_state(self.system_state.translation_state, event)
        elif type(static) is UpFingersEvent and static.num_fingers <= 3:
            self.system_state.set_state(self.system_state.rotation_state, event)
        elif type(static) is UpFingersEvent and static.num_fingers <= 5:
            self.system_state.set_state(self.system_state.scaling_state, event)

    def notify_as_set(self, event):
        pass

class TransformationState(State):
    def __init__(self, system_state) -> None:
        super().__init__(system_state)
        self.reinitialize()

    def reinitialize(self):
        self.motion = False
        self.release = True

class TranslationState(TransformationState):
    def __init__(self, system_state) -> None:
        super().__init__(system_state)

    def apply_event(self, event):

        static, _ = self.get_motion_static_events(event)

        if (type(static) is UpFingersEvent and static.num_fingers <= 3) or type(static) is ClosedHandEvent:
            if self.motion:
                self.system_state.transformation_events_callback(1, static.center)
            elif self.release:
                self.system_state.transformation_events_callback(0, static.center)
                self.system_state.transformation_events_callback(1, static.center)
                self.motion = True
                self.release = False
        elif  (type(static) is UpFingersEvent and static.num_fingers <= 5):
            if self.motion:
                self.system_state.transformation_events_callback(2, static.center)
                self.reinitialize()
            elif self.release:
                   self.reinitialize() 
        elif type(static) is NoEvent:
            if self.motion:
                self.system_state.transformation_events_callback(2,  np.array([0, 0], dtype=np.uint32))
                self.reinitialize()
            elif self.release:
                self.reinitialize()
            self.system_state.selection_callback(0)
        

    def change_state(self, event):
        static, _ = self.get_motion_static_events(event)

        if exit(static):
            self.system_state.set_state(self.system_state.starting_state, event)

    def notify_as_set(self, event):
        pass



class RotationState(TransformationState):
    def __init__(self, system_state) -> None:
        super().__init__(system_state)

    def apply_event(self, event):
        static, _ = self.get_motion_static_events(event)

        if (type(static) is UpFingersEvent and static.num_fingers <= 3) or type(static) is ClosedHandEvent:
            if self.motion:
                self.system_state.transformation_events_callback(1, static.center)
            elif self.release:
                self.system_state.transformation_events_callback(0, static.center)
                self.system_state.transformation_events_callback(1, static.center)
                self.motion = True
                self.release = False
        elif  (type(static) is UpFingersEvent and static.num_fingers <= 5):
            if self.motion:
                self.system_state.transformation_events_callback(2, static.center)
                self.reinitialize()
            elif self.release:
                   self.reinitialize() 
        elif type(static) is NoEvent:
            if self.motion:
                self.system_state.transformation_events_callback(2,  np.array([0, 0], dtype=np.uint32))
                self.reinitialize()
            elif self.release:
                self.reinitialize()
            self.system_state.selection_callback(0)
    def change_state(self, event):
        static, _ = self.get_motion_static_events(event)

        if exit(static):
            self.system_state.set_state(self.system_state.starting_state, event)

    def notify_as_set(self, event):
        pass

class ScalingState(TransformationState):
    def __init__(self, system_state) -> None:
        super().__init__(system_state)

    def apply_event(self, event):
        static, _ = self.get_motion_static_events(event)


        if (type(static) is UpFingersEvent and static.num_fingers <= 3) or type(static) is ClosedHandEvent:
            if self.motion:
                self.system_state.transformation_events_callback(1, static.center)
            elif self.release:
                self.system_state.transformation_events_callback(0, static.center)
                self.system_state.transformation_events_callback(1, static.center)
                self.motion = True
                self.release = False
        elif  (type(static) is UpFingersEvent and static.num_fingers <= 5):
            if self.motion:
                self.system_state.transformation_events_callback(2, static.center)
                self.reinitialize()
            elif self.release:
                   self.reinitialize() 
        elif type(static) is NoEvent:
            if self.motion:
                self.system_state.transformation_events_callback(2,  np.array([0, 0], dtype=np.uint32))
                self.reinitialize()
            elif self.release:
                self.reinitialize()
            self.system_state.selection_callback(0)

    def change_state(self, event):
        static, _ = self.get_motion_static_events(event)

        if exit(static):
            self.system_state.set_state(self.system_state.starting_state, event)

    def notify_as_set(self, event):
        pass

class DrawingState(TransformationState):
    def __init__(self, system_state) -> None:
        super().__init__(system_state)

    def apply_event(self, event):
        static, _ = self.get_motion_static_events(event)

        if (type(static) is UpFingersEvent and static.num_fingers <= 3) or type(static) is ClosedHandEvent:
            if self.motion:
                self.system_state.transformation_events_callback(1, static.center)
            elif self.release:
                self.system_state.transformation_events_callback(0, static.center)
                self.system_state.transformation_events_callback(1, static.center)
                self.motion = True
                self.release = False
        elif  (type(static) is UpFingersEvent and static.num_fingers <= 5):
            if self.motion:
                self.system_state.transformation_events_callback(2, static.center)
                self.reinitialize()
            elif self.release:
                   self.reinitialize() 
        elif type(static) is NoEvent:
            if self.motion:
                self.system_state.transformation_events_callback(2,  np.array([0, 0], dtype=np.uint32))
                self.reinitialize()
            elif self.release:
                self.reinitialize()
            self.system_state.selection_callback(0)


    def change_state(self, event):
        static, _ = self.get_motion_static_events(event)

        if exit(static):
            self.system_state.set_state(self.system_state.starting_state, event)

    def notify_as_set(self, event):
        pass


class ErasingState(TransformationState):
    def __init__(self, system_state) -> None:
        super().__init__(system_state)

    def apply_event(self, event):
        static, _ = self.get_motion_static_events(event)

        if release(static):
            if self.motion:
                self.system_state.transformation_events_callback(2, static.center)
                self.reinitialize()
            elif self.release:
                   self.reinitialize() 
        elif type(static) is ClosedHandEvent:
            if self.motion:
                self.system_state.transformation_events_callback(1, static.center)
            elif self.release:
                self.system_state.transformation_events_callback(0, static.center)
                self.system_state.transformation_events_callback(1, static.center)
                self.motion = True
                self.release = False
        elif exit(static):
            if self.motion:
                self.system_state.transformation_events_callback(2, static.center)
                self.reinitialize()
            elif self.release:
                self.reinitialize()
            self.system_state.selection_callback(0)


    def change_state(self, event):
        static, _ = self.get_motion_static_events(event)

        if exit(static):
            self.system_state.set_state(self.system_state.starting_state, event)

    def notify_as_set(self, event):
        pass


class SavingState(State):
    def __init__(self, system_state) -> None:
        super().__init__(system_state)

    def apply_event(self, event):
        self.system_state.selection_callback(0)

    def change_state(self, event):
        self.system_state.set_state(self.system_state.starting_state, event)
        
    def notify_as_set(self, event):
        pass

