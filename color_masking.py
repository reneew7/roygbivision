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

    lower_blue = np.array([90, 60, 0])
    upper_blue = np.array([121, 255, 255])

    lower_yellow = np.array([25, 70, 120])
    upper_yellow = np.array([30, 255, 255])

    lower_red = np.array([0, 50, 120])
    upper_red = np.array([10, 255, 255])

    lower_green= np.array([40, 70, 80])
    upper_green = np.array([130, 255, 255])

    # portion of an image and will only show the stuff in the range of the color
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask3 = cv2.inRange(hsv, lower_red, upper_red)
    mask4 = cv2.inRange(hsv, lower_green, upper_green)

    # find the contours for each mask => useful for the shape analysis, object detection, & recognition
    contour1 = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour1 = imutils.grab_contours(contour1)

    contour2 = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour2 = imutils.grab_contours(contour2)

    contour3 = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour3 = imutils.grab_contours(contour3)

    contour4 = cv2.findContours(mask4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour4 = imutils.grab_contours(contour4)

    # for each contour, find the area so that any other object will not be in the contour
    # then draw, find the moment, and then label the color with a circle around
    for c in contour1:
        area1 = cv2.contourArea(c)
        if area1 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Blue", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    for c in contour2:
        area1 = cv2.contourArea(c)
        if area1 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Yellow", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    for c in contour3:
        area1 = cv2.contourArea(c)
        if area1 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Red", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    for c in contour4:
        area1 = cv2.contourArea(c)
        if area1 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Green", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)



    # not blue, turn to black
    # result = cv2.bitwise_and(frame, frame, mask=mask)


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