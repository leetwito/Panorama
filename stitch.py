import numpy as np
import cv2
from matplotlib import pyplot as plt


def interpolate(img):
    blurred = cv2.blur(img,(7,7))
    # blurred[img!=0] = img[img!=0]

    # plt.imshow(blurred,cmap='gray')
    # plt.show()
    return blurred




def stitch(left_image, right_image, M):

    leftH,leftW = left_image.shape
    rightH, rightW = right_image.shape

    new_indx = np.zeros((leftH, leftW,2))

    print new_indx.shape



    for r in range(leftH):
        for c in range(leftW):
            new_point = np.dot(M, np.array([c, r, 1]))
            # new_point = new_point/new_point[-1]
            new_indx[r,c,:] = [new_point[1], new_point[0]]

    min_row =  np.min(new_indx[:,:,0]).astype('int')
    min_col =  np.min(new_indx[:, :, 1]).astype('int')

    newH = rightH
    newW = rightW

    if min_row<0:
        new_indx[:, :, 0] = new_indx[:,:,0] + np.abs(min_row)
        newH += np.abs(min_row)
    if min_col<0:
        new_indx[:, :, 1] = new_indx[:, :, 1] + np.abs(min_col)
        newW += np.abs(min_col)




    max_row = np.max(new_indx[:,:,0]).astype('int')
    max_col = np.max(new_indx[:, :, 1]).astype('int')

    # print 'min_row ',min_row
    # print 'min_col ',min_col
    # print 'max_row ',max_row
    # print 'max_col ',max_col

    if max_row > newH:
        newH = max_row
    if max_col > newW:
        newW = max_col


    new_indx = np.floor(new_indx).astype('int')

    # print 'rightH ',rightH
    # print 'rightW ',rightW
    #
    print 'newH ',newH
    print 'newW ',newW


    # new_img = np.zeros((max_row + 1, max_col + 1), dtype='uint8')
    new_img = np.zeros((newH + 1, newW + 1), dtype='uint8')
    # print new_img.shape

    for r in range(leftH):
        for c in range(leftW):
            # g_value = left[r,c]
            # index = new_indx[r,c]
            # new_img[index[0],index[1]] = g_value
            new_img[new_indx[r, c, 0], new_indx[r, c, 1]] = left_image[r, c]
            # new_img[newH -1 - new_indx[r,c,0],new_indx[r,c,1]] = left[r,c]


    new_img = interpolate(new_img)
    # print 'right shape: ',right.shape
    # print 'new image size for right: ',new_img[np.abs(min_row): rightH + np.abs(min_row),
    #                                    np.abs(min_col): rightW + np.abs(min_col)].shape

    new_img[np.abs(min_row): rightH + np.abs(min_row), np.abs(min_col): rightW + np.abs(min_col)] = right_image
    # new_img[-min_row:, -min_col:] = right


    return new_img






# left = cv2.imread('L.jpg', 0)          # queryImage
# right = cv2.imread('R.jpg', 0) # trainImage


# M = np.array([[0,-1,0],[1,0,0],[0,0,1]])
# M = np.array([[np.cos(np.pi/4),-np.sin(np.pi/4),0],[np.sin(np.pi/4),np.cos(np.pi/4),0],[0,0,1]])
# M = np.array([[np.cos(np.pi/6),-np.sin(np.pi/6),0],[np.sin(np.pi/6),np.cos(np.pi/6),0],[0,0,1]])
# M = np.array(
    #[[1 * np.cos(np.pi / 6), -1 * np.sin(np.pi / 6), 0], [np.sin(np.pi / 6), np.cos(np.pi / 6), 0], [0, 0, 1]])
# [[1 * np.cos(np.pi / 6), -1 * np.sin(np.pi / 6), 0], [np.sin(np.pi / 6), np.cos(np.pi / 6), 0], [0, 0, 1]])
# M = np.array([[1,0,0],[0,1,0],[0,0,1]])

# M = np.array([[  3.22540345e-01 , -1.29091426e-01 , -1.41297678e+02],
#  [  4.50126539e-01 , -1.03096664e+00 ,  7.04625196e+02],
#  [  5.79094738e-04  ,-1.45585410e-03 ,  1.00000000e+00]])

# M = np.linalg.inv(M)

# M = -M

# panorama = warp(left,right,M)
# panorama = warp(right,left,M)

# plt.imshow(panorama, cmap='gray')
# plt.show()

# cv2.imshow('panorama',cv2.resize(panorama, (int(0.5*panorama.shape[1]), int(0.5*panorama.shape[0]))) )
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()





#
# himg2, wimg2 = left.shape
# right = cv2.resize(right, (wimg2, himg2))
# # print(left.shape, right.shape)
#
# # Initiate SIFT detector
# orb = cv2.ORB_create()
#
# # find the keypoints and descriptors with SIFT
# kpl, desl = orb.detectAndCompute(left, None)
# kpr, desr = orb.detectAndCompute(right, None)
#
# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# # Match descriptors.
# matches = bf.match(desl, desr)
#
# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
#
# left_pts = np.float32([kpl[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
# right_pts = np.float32([kpr[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
#
#
# M, mask = cv2.findHomography(left_pts, right_pts, cv2.RANSAC, 5.0)
# # matchesMask = mask.ravel().tolist()
# # print M
#
# himg2, wimg2 = right.shape
# img3_size = (himg2, wimg2)
# h,w = img1.shape
# pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
# dst = cv2.warpPerspective(img1, M, img3_size)

# print dst.shape
# plt.imshow(dst),plt.show()

# result = cv2.warpPerspective(right, M,
#                              (left.shape[1] + right.shape[1] + 200, left.shape[0] +right.shape[0]))

# left_warped = cv2.warpPerspective(left, M,(left.shape))

# result[0:img2.shape[0], 0:img2.shape[1]] = img2
# result[0:right.shape[0], left.shape[1]:] = right

# img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

# Draw first 10 matches.
# img3 = cv2.drawMatches(left, kpl, right, kpr, matches[:10], outImg=None, flags=2)

# plt.imshow(left_warped,cmap='gray'),plt.show()