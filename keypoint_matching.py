##
# cording:utf-8
##

import numpy as np
import cv2

def main():
    print('keypoint matching')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        exit()

    ret, bimg = cap.read()

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    while(1):
        ret, aimg = cap.read()
        if not ret:
            break

        gbimg = cv2.cvtColor(bimg, cv2.COLOR_BGR2GRAY)
        gaimg = cv2.cvtColor(aimg, cv2.COLOR_BGR2GRAY)

        # ORBでキーポイントを計算
        kp1 = orb.detect(gbimg, None)
        kp2 = orb.detect(gaimg, None)

        # ORBで特徴記述子を計算
        kp1, des1 = orb.compute(gbimg, kp1)
        kp2, des2 = orb.compute(gbimg, kp2)

        if des1 is None or des2 is None:
            continue

        if len(des1) < 15 or len(des2) < 15:
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 10 matches.
        mimg = cv2.drawMatches(bimg, kp1, aimg, kp2, matches[:15], None, flags=2)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:15]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:15]]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        print(M)

        cv2.imshow('img', mimg)
        key = cv2.waitKey(1)
        if key == 27:
            break

        bimg = aimg.copy()

if __name__ == '__main__':
    main()