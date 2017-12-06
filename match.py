import numpy as np
import cv2


def l2_metric(vec1, vec2):
    return np.sqrt(sum((vec1 - vec2)**2))


def hamming_metric(vec1, vec2):
    return np.count_nonzero(vec1 != vec2)


def match(kp1, kp2, des1, des2, distance_func, n = None):
    """

    :param des1: descriptors of the src image - size nXm where n is number of keypoints and m number of features.
    :param des2: descriptors of the dst image - size wXm where w is number of keypoints and m number of features.
    :param distance_func: the function that calculate the distance between two feature vectors.
    :param n: Optional - number of best matches to return, by default min(n,w)/2
    :return: array of length n - of cv2.DMatch objects (https://docs.opencv.org/3.3.1/d4/de0/classcv_1_1DMatch.html):
                                    dmatch.queryIdx is the index of the keypoint of src image
                                    dmatch.trainIdx is the index of the keypoint of dst image
                                    dmatch.distance is the distance between the two keypoints
    """
    if n is None:
        n = min(des1.shape[0], des2.shape[0])
    else:
        n = min(n, min(des1.shape[0], des2.shape[0]))

    dist = np.zeros((des1.shape[0], des2.shape[0]))
    for i in range(des1.shape[0]):
        for j in range(des2.shape[0]):
            dist[i,j] = distance_func(des1[i,:], des2[j,:])
### DANI AND RAFI FOUND PROBLEM: Function may match one kp to several others, should match each kp to just 1 kp

    flatten_dist = dist.flatten()
    n_smallest_indices = flatten_dist .argsort()[:n]
    x, y = np.unravel_index(n_smallest_indices, dist.shape)
################################################
    n_smallest_values = flatten_dist[n_smallest_indices]  # n smallest distances

    best_match = []
    for i in range(n):
        best_match.append([kp1[x[i]].pt[0], kp1[x[i]].pt[1], kp2[y[i]].pt[0], kp2[y[i]].pt[1]])

    return np.array(best_match)
