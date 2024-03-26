import sys
import os
import math

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from .base_frame_extractor import *
from .video_validation import *
from pipeline_processor.record import *
from .gridview_generator import *


class FpsExtractor(BaseFrameExtractor):
    def __init__(self, video_path):
        super().__init__(video_path)

    def __del__(self):
        self._release_video()

    def extract_frames(self, **kwargs):
        self._extract_arguments(**kwargs)
        self._open_video(self.ts)
        self.image_data, self.frame_count = self._process_video()
        return self.image_data

    def _open_video(self, ts=None):
        self.video_loader = VideoLoader(self.video_path)

        (
            self.video_capture,
            self.fps,
            self.total_frames,
            self.video_length,
        ) = self.video_loader.open_video(ts=ts)

    def _release_video(self):
        self.video_loader.release_video()

    def _extract_arguments(self, **kwargs):
        try:
            self.frame_fixed_number = kwargs.get("frame_fixed_number", 6)
            self.ts = kwargs.get("ts", None)
        except Exception as e:
            raise Exception(e)

    def _process_video(self):
        frame_count = 0
        output_frame_index = 0
        list_frame_data = []

        self._adjust_interval_fixed()

        frames = self.video_capture.iter_frames()

        for frame in frames:

            if frame_count % self.frames_per_interval == 0:
                start = output_frame_index * self.frames_per_interval
                frame_index_selected = start

            if frame_index_selected == frame_count:
                list_frame_data.append(frame)
                output_frame_index += 1

            if len(list_frame_data) == self.frame_fixed_number:
                break

            if frame_count == self.total_frames:
                break

            frame_count += 1

        return list_frame_data, output_frame_index

    def _adjust_interval_fixed(self):
        self.frames_per_interval = math.floor(
            self.total_frames / self.frame_fixed_number
        )


def main():
    path = ["example", "ysTmUTQ5wZE_17_45.mp4"]
    tmp = FpsExtractor(path)
    print(tmp.video_path)
    tmp.save_data_based_on_option(
        SaveOption.FILE,
        filename="/example/extraction_sample/ysTmUTQ5wZE_17_45/",
        frame_fixed_number=6,
    )


if __name__ == "__main__":
    main()
