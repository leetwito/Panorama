import cv2
import numpy as np
import math
import matplotlib as plt
import colorsys

# Part A (Rafael and Maor part):
import MyHarrisCornerDetector as HCD

# Part B (Shahar and Reut): implementing the match function
import match

# Part C (Lee and Daniel):
import findHomography
# Part D (Ira and Ofer):


def main():
    src_image_path = "src.jpg"
    des_image_path = "dst.jpg"

    img1 = cv2.imread(src_image_path, 0) / 255.0  # greyscale
    img2 = cv2.imread(des_image_path, 0) / 255.0  # greyscale

    # Part A
    corner_points1, descriptors1 = HCD.harris_corner(img1)
    corner_points2, descriptors2 = HCD.harris_corner(img2)

    print "descriptors1 length ", len(descriptors1)
    print "descriptors2 length ", len(descriptors2)

    # Part B
    matches_arr = match.match(corner_points1, corner_points2, descriptors1, descriptors2, match.hamming_metric, n=50)
    print matches_arr.shape
    print matches_arr
    

    # Part C
    H=findHomography.best_H(matches_arr)
    print H
    # Part D





if __name__=="__main__":
    main()
