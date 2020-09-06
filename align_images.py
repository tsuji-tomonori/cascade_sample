import argparse
from pathlib import Path

import cv2

from util import detect

def act(img, cascade, h, w):
    positions = detect.detect(img, cascade)
    positions = detect.zoom_out(positions)
    imgs = detect.cut(img, positions)
    imgs = [cv2.resize(img, (h, w)) for img in imgs]
    return imgs

def main():
    parser = argparse.ArgumentParser(description='Detects the faces and aligns the images using a cascade file')
    parser.add_argument('src_dir', help='Directory with aligned images for projection')
    parser.add_argument('dst_dir', help='Output directory')
    parser.add_argument('--xml_dir', default='xml\lbpcascade_animeface.xml', help='Temporary directory for cascade file')
    parser.add_argument('--img_hight', type=int, default='1024', help='Image hight')
    parser.add_argument('--img_width', type=int, default='1024', help='Image width')
    args = parser.parse_args()

    SUFFIX = [".jpg", ".png"]

    # set file path
    src_files = [x for x in Path(args.src_dir).iterdir() if x.suffix in SUFFIX]
    # load cascade file
    cascade = cv2.CascadeClassifier(str(args.xml_dir))

    # main processing
    for file in src_files:
        input_img = cv2.imread(str(file))
        output_imgs = act(input_img, cascade, args.img_hight, args.img_width)
        for idx, img in enumerate(output_imgs):
            # set file path
            file_name = file.parts[-1].split(".")
            file_name = file_name[0] + '_' + str(idx) + '.' + file_name[1]
            output_file = Path(args.dst_dir) / file_name
            # save img
            cv2.imwrite(str(output_file), img)

if __name__ == "__main__":
    main()