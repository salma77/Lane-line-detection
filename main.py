import numpy as np
from moviepy.editor import VideoFileClip
from lane_lines import *
import sys


def main():
    args = sys.argv[1:]
    if len(args) == 0:
        print("No arguments provided")
        return
    if len(args) == 1:
        debug = True
        img = args[0]
        out_img = lane_finding_pipeline(img, debug)
    else:
        video_output = args[1]
        clip1 = VideoFileClip(args[0])
        output_clip = clip1.fl_image(lane_finding_pipeline)
        output_clip.write_videofile(video_output, audio=False)

if __name__ == '__main__':
    main()
