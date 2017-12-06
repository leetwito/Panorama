import numpy as np
import cv2

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "({}, {})".format(self.x, self.y)


def findHomography(points1, points2):
    A = np.zeros((len(points1) * 2, 9))
    i = 0
    for p1, p2t in zip(points1, points2):
        ax = np.array([-p1.x, -p1.y, -1, 0, 0, 0, p2t.x * p1.x, p2t.x * p1.y, p2t.x])
        ay = np.array([0, 0, 0, -p1.x, -p1.y, -1, p2t.y * p1.x, p2t.y * p1.y, p2t.y])
        A[i, :] = ax
        A[i + 1, :] = ay
        i += 2
    return A


def genPoint(p1, H):
    p_calc = np.array([[p1.x], [p1.y], [1]])
    point2_gen = np.dot(H, p_calc)
    if point2_gen[2] != 0:
        return [point2_gen[0] / point2_gen[2], point2_gen[1] / point2_gen[2]]
    else:
        raise ZeroDivisionError

def choosePoints(all_points1, all_points2):
    indices = np.random.choice(range(len(all_points1)), 4, replace=False)
    points1 = [all_points1[i] for i in indices]
    points2 = [all_points2[i] for i in indices]
    return (points1, points2)

def calcH(points1, points2):
    A = findHomography(points1, points2)
    #print A.shape
    h = solveAh(A)
    H = np.reshape(h, (3, 3))
    return H

def pointsDistance(pt, p2):
    return np.sqrt((pt[0] - p2.x) ** 2 + (pt[1] - p2.y) ** 2)


def Ransac(all_points1, all_points2):
    min_MSE = None
    min_H = None
    for k in range(2000):
        points1, points2 = choosePoints(all_points1, all_points2)
        H = calcH(points1, points2)
        MSE = 0
        for p1, p2 in zip(all_points1, all_points2):
            point2_t = genPoint(p1, H)
            cur_MSE = pointsDistance(point2_t, p2)
            MSE += cur_MSE
        if MSE < min_MSE or min_MSE is None:
            min_MSE = MSE
            min_H = H
    print ('we have {} point.\navg MSE before removing outliers:{}').format(len(all_points1), min_MSE/len(all_points1))
    return min_H/min_H[2,2]


def removeOutliers(H,points1,points2):
    MSE=[]
    for p1,p2 in zip(points1,points2):
        pt=genPoint(p1, H)
        MSE.append(pointsDistance(pt, p2))
    thresh=12
    points1 = np.array(points1)
    points1_new =[po1 for x,po1 in enumerate(points1) if MSE[x]<thresh]
    points2_new =[po2 for x,po2 in enumerate(points2) if MSE[x]<thresh]

    return  points1_new, points2_new


def solveAh(A):
    U, S, V = np.linalg.svd(A)
    i = len(S) - 1
    V = V.T
    h = V[:, i]
    return h

def best_H(matches_arr):
    points1, points2 = [], []

    for vals in matches_arr:

        points1.append(Point(int(vals[0]), int(vals[1])))
        points2.append(Point(int(vals[2]), int(vals[3])))
    min_H = Ransac(points1, points2)
    points1_new, points2_new = removeOutliers(min_H, points1, points2)
    min_H_rem_out = calcH(points1_new, points2_new)
    min_H_rem_out = min_H_rem_out / min_H_rem_out[2, 2]
    MSE = 0
    for p1, p2 in zip(points1_new, points2_new):
        point2_t = genPoint(p1, min_H_rem_out)
        MSE += pointsDistance(point2_t, p2)
    print "amount of clean points:{} MSE:{}".format(len(points1_new),MSE/len(points1_new))
    return min_H_rem_out


def main():
    f = open('match.txt', 'r').read()
    lines = f.splitlines()
    matches_arr=[]
    for line in lines:
        vals = line.split(',')
        vals=map(int,vals)
        matches_arr.append(vals)
    #matches_arr = map(int, matches_arr)

    print best_H(matches_arr)


if __name__ == '__main__':
    main()
