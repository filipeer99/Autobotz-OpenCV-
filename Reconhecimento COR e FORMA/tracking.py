import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    cnt_blue = 0
    cnt_red = 0
    # COLOR -> U have to test using track bars first
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)  # noise
    # RED
    low_red = np.array([0, 130, 79])
    high_red = np.array([7, 255, 255])
    red_mask = cv2.inRange(hsv, low_red, high_red)
    red_mask = cv2.erode(red_mask, kernel)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)
    # BLUE
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv, low_blue, high_blue)
    blue_mask = cv2.erode(blue_mask, kernel)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
    # Shape -> Square (contours)
    contours_1, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Red detected
    contours_2, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Blue detected

    for cnt in contours_1:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        x1 = approx.ravel()[0]
        y1 = approx.ravel()[1]
        if area > 200:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
            cnt_blue += 1
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspectRatio = float(w) / h
                if 0.90 <= aspectRatio <= 1.10:
                    cv2.putText(frame, "Red Square", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            if 10 < len(approx) < 20:  # A circle is detected approx with 10 to 20 sides (can be done with hough
                # transformation too)
                cv2.putText(frame, "Red Circle", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    for cnt in contours_2:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        x1 = approx.ravel()[0]
        y1 = approx.ravel()[1]
        if area > 100:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
            cnt_red += 1
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspectRatio = float(w) / h
                if 0.90 <= aspectRatio <= 1.10:
                    cv2.putText(frame, "Blue Square", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
            if 10 < len(approx) < 20:
                cv2.putText(frame, "Blue Circle", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))

    cv2.imshow("Frame", frame)
    # cv2.imshow("Red", red_mask)
    # cv2.imshow("Blue", blue_mask)

    print("Blue: " + str(cnt_blue) + ' '"Red: " + str(cnt_red))

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
