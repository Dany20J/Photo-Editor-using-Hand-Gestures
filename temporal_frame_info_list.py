
class TemporalFrameInfoList:
    def __init__(self) -> None:
        self.frame_infos = []
        self.if_consumed_frames = []

    def add_frame(self, frame_info):
        self.frame_infos.append(frame_info)
        self.if_consumed_frames.append(False)

    def consume_frame(self, frame_num):
        if len(self.if_consumed_frames) != 0:
            self.if_consumed_frames[frame_num] = True

    def release_consumed_frames(self):
        new_frame_infos = []
        new_if_consumed_frames = []

        for if_consumed_frame, frame_info in self.reversed_iterator():
            if if_consumed_frame:
                break
            else:
                new_frame_infos.append(frame_info)
                new_if_consumed_frames.append(False)
        self.frame_infos = new_frame_infos

        self.if_consumed_frames = new_if_consumed_frames

    def reversed_iterator(self):
        return reversed(list(zip(self.if_consumed_frames, self.frame_infos)))

    def modify_as_consumed_from_end(self, num):
        if num >= 1:
            self.if_consumed_frames[-1] = True
            return
        for ind in range(self.__len__() - 1, self.__len__() - num - 1, -1):
            self.if_consumed_frames[ind] = True

    def __len__(self):
        return len(self.frame_infos)

    def __str__(self) -> str:
        strig = ""
        for frame_info in self.frame_infos:
            strig += str(frame_info) + ","
        return strig
