import cv2
import numpy as np
import math
import matplotlib as plt
import colorsys


def derive_image(image):
    filter_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    filter_y = filter_x.T
    I_x = np.abs(cv2.filter2D(image, -1, filter_x))
    I_y = np.abs(cv2.filter2D(image, -1, filter_y))
    return I_x, I_y
    # cv2.imshow("Ix", I_x)
    # cv2.waitKey()
    # cv2.imshow("Iy", I_y)
    # cv2.waitKey()


def find_corners(Ix, Iy, K, TH):
    corners_mat = np.zeros(Ix.shape)

    for row in range(2, corners_mat.shape[0]-1):
        for col in range(2, corners_mat.shape[1] - 1):
            window_x = Ix[row - 2:row + 2, col - 2:col + 2] #5X5
            window_y = Iy[row - 2:row + 2, col - 2:col + 2]  # 5X5

            Ix_2 = sum(sum(window_x ** 2))
            Iy_2 = sum(sum(window_y ** 2))
            Ix_Iy = sum(sum(window_x*window_y))
            harris_mat = np.array([[Ix_2, Ix_Iy],[Ix_Iy, Iy_2]])

            det = np.linalg.det(harris_mat)
            tr = np.matrix.trace(harris_mat)
            corners_mat[row, col] = det-K*tr**2
 #   print corners_mat
    thresh_harris_mat = np.abs(corners_mat) > TH
    return thresh_harris_mat


def descriptor(corner_points, image):
    orb = cv2.ORB_create()
    corner_points, descriptors = orb.compute(image, corner_points)
    return corner_points, descriptors

def harris_corner(img):
    # img = cv2.imread('chess.jpg', 0)/255.0 #greyscale
    TH = 1
    Ix, Iy = derive_image(img)
    # corner_mat = find_corners(Ix, Iy, 0.04, 200)
    corner_mat = find_corners(Ix, Iy, 0.04, TH)
    # TH=0.01- colouds,  0.1- montain
    corner_points = corner_mat.nonzero()
    corner_keys = []
    for index in range(corner_points[0].shape[0]):
        corner_keys.append(cv2.KeyPoint(corner_points[0][index], corner_points[1][index], 5))
    #corner_points = cv2.KeyPoint(corner_mat.)
    corner_mat_nums = corner_mat.astype(int)*255.0
    corner_points, descriptors = descriptor(corner_keys, img.astype('uint8'))

    # cv2.imshow("corner mat", corner_mat_nums.astype('uint8'))
    # cv2.waitKey()

    return corner_points, descriptors

    #cv2.imshow("corner mat", corner_mat_nums.astype('uint8'))
    #cv2.waitKey()

def main():
    img1 = cv2.imread('src.jpg', 0) / 255.0  # greyscale
    corner_points1, descriptors1 = harris_corner(img1)

    img2 = cv2.imread('dest.jpg', 0) / 255.0  # greyscale
    corner_points2, descriptors2 = harris_corner(img2)

if __name__=="__main__":
    main()
