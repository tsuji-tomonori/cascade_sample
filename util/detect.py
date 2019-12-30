import cv2
import numpy


def detect(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    positions = cascade.detectMultiScale(gray,
                                         # detector options
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(32, 32))
    return positions


def cut(img, positions):
    return [img[y:y+h, x:x+w] for x, y, w, h in positions]


def draw_bounding_box(img, positions):
    for x, y, w, h in positions:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return img

def zoom_out(positions, scale=1.5):
    ret_position = []
    for x, y, w, h in positions:
        w_new = int(w * scale)
        h_new = int(h * scale)
        x = max(0, int(x - ((w_new - w) / 2)))
        y = max(0, int(y - ((h_new - h) / 2)))
        ret_position.append((x, y, w_new, h_new))
    return ret_position