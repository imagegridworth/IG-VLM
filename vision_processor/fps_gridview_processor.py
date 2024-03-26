import sys
import os
import math
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from pipeline_processor.record import *
from .fps_extractor import *
from .gridview_generator import *
from .video_validation import *


class FpsDataProcessor:
    def __init__(
        self,
        calcualte_max_row=lambda x: round(math.sqrt(x)),
        save_option=SaveOption.IMAGE,
        frame_fixed_number=6,
    ):
        self.calculate_max_row = calcualte_max_row
        self.frame_fixed_number = frame_fixed_number
        self.save_option = save_option

    def process(self, video_path, ts=None):
        fps_extractor = FpsExtractor(video_path)
        grid_view_creator = GridViewCreator(
            self.calculate_max_row,
        )

        try:
            rlt_fps_extractor = fps_extractor.save_data_based_on_option(
                SaveOption.NUMPY,
                frame_fixed_number=self.frame_fixed_number,
                ts=ts,
            )
            rlt_grid_view_creator = grid_view_creator.post_process_based_on_options(
                self.save_option, rlt_fps_extractor
            )
        except Exception as e:
            print("Exception : %s on %s" % (str(e), str(video_path)))
            return -1

        return rlt_grid_view_creator


def main():

    video_name = "rlQ2kW-FvMk_66_79.mp4"

    fps_data_processor = FpsDataProcessor(
        save_option=SaveOption.IMAGE,
        frame_fixed_number=6,
    )
    print(vars(fps_data_processor))
    rlt = fps_data_processor.process(["example", video_name])
    print(rlt)

    rlt.save("./example/imagegrid_sample/%s.jpg" % (video_name.split(".")[0]))


if __name__ == "__main__":
    main()
