import numpy as np
import cv2
import imutils

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    # takes RGB pixel values to HSV values
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_bounds = dict()

    color_bounds["Red"] = np.array([[0, 60, 120], [10, 255, 255]])
    color_bounds["Orange"] = np.array([[11, 60, 80], [25, 255, 255]])
    color_bounds["Yellow"] = np.array([[26, 60, 120], [35, 255, 255]])
    color_bounds["Green"] = np.array([[36, 60, 80], [70, 255, 255]])
    color_bounds["Blue"] = np.array([[71, 60, 0], [120, 255, 255]])
    color_bounds["Purple"] = np.array([[121, 60, 80], [1400, 255, 255]])
    color_bounds["Pink"] = np.array([[141, 60, 80], [179, 255, 255]])
    color_bounds["White"] = np.array([[0, 0, 200], [179, 60, 255]])
    color_bounds["Black"] = np.array([[0, 0, 0], [179, 60, 50]])
    color_bounds["Brown"] = np.array([[0, 0, 120], [10, 59, 119]])

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

                # color = min_color(patch[0], patch[1], patch[2])
                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                color = colors[i]
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



