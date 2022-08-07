import cv2
import numpy as np
import pywavefront


def rotate_rvec(rvec, alpha):
    res = np.ndarray(shape=(1, 1, 3), dtype=np.float64)
    res[0][0][0] = rvec[0][0][0] + alpha
    res[0][0][1] = rvec[0][0][1]
    res[0][0][2] = rvec[0][0][2]
    return res


def check_rvec(rvec):
    if rvec.shape != (1, 1, 3):
        print("DETECTED")
        res = np.ndarray(shape = (1, 1, 3) ,dtype=np.float64)

        res[0][0][0] = rvec[0][0][0]
        res[0][0][1] = rvec[0][0][1]
        res[0][0][2] = rvec[0][0][2]
        return res
    else:
        return rvec


def check_tvec(tvec):
    if tvec.shape != (1, 1, 3):
        print("DETECTED T")
        res = np.ndarray(shape = (1, 1, 3) ,dtype=np.float64)

        res[0][0][0] = tvec[0][0][0]
        res[0][0][1] = tvec[0][0][1]
        res[0][0][2] = tvec[0][0][2]
        return res
    else:
        return tvec

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    arucoParams = cv2.aruco.DetectorParameters_create()

    cameraMatrix = np.array([[8.2177218160147447e+02, 0., 3.2094289806342221e+02],
                             [0., 8.2177218160147447e+02, 2.4095144956871457e+02],
                             [0., 0., 1.]])
    dist = np.array([-2.29320995e-02,  2.63169501e+00,
                     -3.33186134e-03,  1.57736647e-02,
                     -1.71779274e+01])

    bunny = pywavefront.Wavefront('bunny.obj')
    vertices = np.array(bunny.vertices, dtype=np.float32)

    alpha = 0.001

    while True:
        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (corners, ids, rejected) = cv2.aruco.detectMarkers(gray,
                                                           arucoDict, parameters=arucoParams)

        if corners:
            for (markerCorner, markerID) in zip(corners, ids):
                try:
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, dist)
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners

                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    cv2.line(img, topLeft, topRight, (0, 255, 0), 2)
                    cv2.line(img, topRight, bottomRight, (0, 255, 0), 2)
                    cv2.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv2.line(img, bottomLeft, topLeft, (0, 255, 0), 2)

                    rvec = check_rvec(rvec)
                    tvec = check_tvec(tvec)

                    img = cv2.aruco.drawAxis(img, cameraMatrix, dist, rvec, tvec, 0.1)

                    # rotation example
                    # rvec = rotate_rvec(rvec, alpha)
                    # alpha += 0.1

                    projected_vertices, _ = cv2.projectPoints(vertices,
                                                              rvec,
                                                              tvec,
                                                              cameraMatrix,
                                                              dist)

                    for vertex in projected_vertices:
                        try:
                            x = int(vertex[0][0])
                            y = int(vertex[0][1])
                            if 0 < x < 640 and 0 < y < 480:
                                cv2.circle(img, [x, y], 1, (0, 255, 0))
                        except Exception as e:
                            pass
                except Exception as e:
                    print(e)

        cv2.imshow("img", img)
        cv2.waitKey(1)
