import numpy as np
import cv2
import imutils
import pandas as pd

cap = cv2.VideoCapture(0)


def min_color(r, g, b):
    """
    :param r: R
    :param g: G
    :param b: B
    :return: closest color from dataset
    """
    error = np.infty
    idx = 0
    for col in range(df.shape[0]):
        red = df.loc[col, 'r']
        gr = df.loc[col, 'g']
        blue = df.loc[col, 'b']
        r_e = (red - r) ** 2
        g_e = (gr - g) ** 2
        b_e = (blue - b) ** 2
        total_e = r_e + g_e + b_e
        if total_e < error:
            idx = col
            error = total_e
    return df.loc[idx, 'formatted_name']


while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    # takes RGB pixel values to HSV values
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    df = pd.read_csv('colors.csv')
    df.columns = ['name', 'formatted_name', 'hex', 'r', 'g', 'b']
    color_bounds = dict()
    """
    for i in range(len(df)):
        name = df.loc[i, 'formatted_name']
        r = df.loc[i, 'r']
        g = df.loc[i, 'g']
        b = df.loc[i, 'b']
        lower = np.array([max(0, r - 5), max(0, r - 5), max(0, r - 5)])
        upper = np.array([min(255, r + 5), min(255, r + 5), min(255, r + 5)])
        color_bounds[name] = np.array([lower, upper])
    """

    color_bounds["Blue"] = np.array([[90, 60, 0], [121, 255, 255]])
    color_bounds["Yellow"] = np.array([[25, 70, 120], [30, 255, 255]])
    color_bounds["Red"] = np.array([[0, 50, 120], [10, 255, 255]])
    color_bounds["Green"] = np.array([[40, 70, 80], [130, 255, 255]])

    colors = list(color_bounds.keys())
    masks = list()
    for color in colors:
        mask = cv2.inRange(hsv, color_bounds[color][0, :], color_bounds[color][1, :])
        masks.append(mask)

    contours = list()
    for mask in masks:
        contour = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = imutils.grab_contours(contour)
        contours.append(contour)

    for i in range(len(contours)):
        for c in contours[i]:
            area = cv2.contourArea(c)
            if area > 5000:
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
                M = cv2.moments(c)

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                patch = np.zeros((100, 3))
                for j in range(-5, 5):
                    for k in range(-5, 5):
                        indx = (j + 5) * 10 + (k + 5)
                        patch[indx, :] = frame[cy + j, cx + k]
                patch = np.average(patch, axis=0)

                color = min_color(patch[0], patch[1], patch[2])
                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                # color = colors[i]
                cv2.putText(frame, color, (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    cv2.imshow('result', frame)

    # if cv2.waitKey(1) == ord('q'):
    #     break

    # wait for frame to refresh
    k = cv2.waitKey(5)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

# get different colors
# BGR_color = np.array([[[255, 0, 0]]])
# x = cv2.cvtColor(BGR_color, cv2.COLOR_BGR2HSV)
# x[0][0]



