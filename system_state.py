

import gesture_state
# from tkGUI import App


class DictEvent(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class SystemState:

    def __init__(self, gui) -> None:
        self.gui = gui
        # self.gui = App()
        self.selections = {0: "", 1: "translate",
                           2: "rotate", 3: "scale", 4: "paint",
                           5: "erase", 6: "save"}
        self.mouse_events = {0: "ButtonPress", 1: "Motion", 2: "ButtonRelease"}

        self.starting_state = gesture_state.StartingState(self)

        self.translation_state = gesture_state.TranslationState(self)
        self.rotation_state = gesture_state.RotationState(self)
        self.scaling_state = gesture_state.ScalingState(self)
        self.drawing_state = gesture_state.DrawingState(self)
        self.erasing_state = gesture_state.ErasingState(self)
        self.loading_state = gesture_state.SavingState(self)

        self.current_state = self.starting_state

        self.selection_callback(0)

    def selection_callback(self, event_selection_num):
        if event_selection_num <= 5:
            self.gui.saved_image = False
            self.gui.setSelectedOperationCallback(
                self.selections[event_selection_num])
        else:
            if event_selection_num == 6:
                if not self.gui.saved_image:
                    self.gui.saveImageCallback()
                

    def transformation_events_callback(self, mouse_event_num, center):
        event = DictEvent()
        event.type = self.mouse_events[mouse_event_num]
        event.x = int(center[0])
        event.y = int(center[1])
        event.x, event.y = event.y, event.x
        print(event)

        self.gui.guiEventsCallback(event)



    def set_state(self, state, event=None):
        self.current_state = state
        self.current_state.notify_as_set(event)
        print("new state", self.current_state)

    def apply_event(self, event):

        self.current_state.apply_event(event)
        self.current_state.change_state(event)


