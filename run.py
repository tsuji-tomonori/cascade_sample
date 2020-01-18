import os
import cv2
from pathlib import Path

from util import workdir, detect

CASCADE_NAME = "lbpcascade_animeface.xml"
MODE = 1
SUFFIX = [".jpg", ".png"]

# set path
workdir = workdir.workdir()
read_dir = workdir / "input"
read_file_names = [x.name for x in read_dir.iterdir() if x.suffix in SUFFIX]
read_pathes = [read_dir / name for name in read_file_names]
write_dir = workdir / "output"
write_pathes = [write_dir / name for name in read_file_names]
cascade_file = workdir / "xml" / CASCADE_NAME

# load cascade file
cascade = cv2.CascadeClassifier(str(cascade_file))

# main processing
for read, write in zip(read_pathes, write_pathes):
    # input
    img = cv2.imread(str(read), cv2.IMREAD_COLOR)
    # detect
    positions = detect.detect(img, cascade)
    # zoom out
    #positions = detect.zoom_out(positions, scale=1.5)
    # output
    # cut
    if MODE == 0:
        imgs = detect.cut(img, positions)
        for i, img in enumerate(imgs):
            # set file path
            file = write.parts[-1].split(".")
            file = file[0] + "#" + str(i) + "." + file[1]
            file = str(Path(*write.parts[:-1], file))
            # save img
            cv2.imwrite(file, img)

    # draw bounding box
    if MODE == 1:
        img = detect.draw_bounding_box(img, positions)
        # save img
        cv2.imwrite(str(write), img)
