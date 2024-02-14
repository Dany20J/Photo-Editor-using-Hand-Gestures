from abc import ABC
import numpy as np

class Event(ABC):
    def __init__(self) -> None:
        super().__init__()


class LeftToRightLineMotionEvent(Event):
    def __init__(self) -> None:
        super().__init__()

class RightToLeftLineMotionEvent(Event):
    def __init__(self) -> None:
        super().__init__()

class TopToBottomLineMotionEvent(Event):
    def __init__(self) -> None:
        super().__init__()

class BottomToTopLineMotionEvent(Event):
    def __init__(self) -> None:
        super().__init__()


class ClockwiseCircularMotionEvent(Event):
    def __init__(self) -> None:
        super().__init__()

class CounterClockwiseCircularMotionEvent(Event):
    def __init__(self) -> None:
        super().__init__()

class HandEvent(Event):
    def __init__(self, center=None) -> None:
        super().__init__()
        self.center = np.copy(center)

class ClosedHandEvent(HandEvent):
    def __init__(self, center=None) -> None:
        super().__init__(center)
    def __str__(self) -> str:
        return "ClosedHandEvent"

class UpFingersEvent(HandEvent):
    def __init__(self, center, num_fingers) -> None:
        super().__init__(center)
        self.num_fingers = num_fingers
    def __str__(self) -> str:
        return f"UpFingersEvent_{self.num_fingers}"

class NoEvent(Event):
    def __init__(self) -> None:
        super().__init__()
    def __str__(self) -> str:
        return "NoEvent"

class NoDecisionYetEvent(Event):
    def __init__(self) -> None:
        super().__init__()
    def __str__(self) -> str:
        return "NoDecisionYetEvent"