import train
import test

import matplotlib.pyplot as plt
import numpy as np
import cv2

CIRCLE_STORKE = 1

sketchbook_img = np.zeros((28, 28, 3), np.uint8)
draw_flag = False
ix, iy = -1, -1

# train_module = train.Train()
# train_module.train()

test_module = test.Test()

def draw_brush(event, x, y, flags, params):
    global ix, iy, draw_flag

    if event == cv2.EVENT_LBUTTONDOWN:
        draw_flag = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        draw_flag = False
        cv2.circle(sketchbook_img, (x, y), CIRCLE_STORKE, (255, 255, 255), -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if draw_flag:
            cv2.circle(sketchbook_img, (x, y), CIRCLE_STORKE, (255, 255, 255), -1)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_brush)

while True:
    cv2.imshow("image", sketchbook_img)

    gray_sketch_img = cv2.cvtColor(sketchbook_img, cv2.COLOR_BGR2GRAY)

    if cv2.waitKey(10) & 0xFF == ord('t'):
        convs = test_module.test(gray_sketch_img)
        
        plt.subplot(1, len(convs), 1)
        for i in range(len(convs)):
            plt.subplot(1, len(convs), i + 1)
            plt.imshow(convs[i], cmap='gray')
        plt.show()

    if cv2.waitKey(10) & 0xFF == ord('c'):
        sketchbook_img = np.zeros((28, 28, 3), np.uint8)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
