import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

src_path = 'input'
dst_path = 'output'
src_files = os.listdir(src_path)
dst_files = os.listdir(dst_path)

k_mtx = np.array([[3980.493078255596, 0, 1986.461800767714], [0, 3976.316218026821, 1438.517307755756], [0, 0, 1]])
DistCoef = np.array([-0.505051155821860, 0.353517378786100, 6.004038559166109e-04, 3.813929798209635e-04, -0.204032578999073])

def main():
    i = 0
    while i < len(src_files):
        src = os.path.join(src_path, src_files[i])
        img = cv2.imread(src)

        img = undistort(img, k_mtx, DistCoef, verbose=False)

        cv2.imwrite(dst_path + '/' + 'undist' + src_files[i], img)

        print(src_files[i])

        i += 1
        if i == len(src_files):
            break

    print('All files are pre-annotatted')                

def undistort(frame, mtx, dist, verbose=False):
    """
    Undistort a frame given camera matrix and distortion coefficients.
    :param frame: input frame
    :param mtx: camera matrix
    :param dist: distortion coefficients
    :param verbose: if True, show frame before/after distortion correction
    :return: undistorted frame
    """
    frame_undistorted = cv2.undistort(frame, mtx, dist, newCameraMatrix=mtx)

    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax[1].imshow(cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB))
        plt.show()

    return frame_undistorted

if __name__ == '__main__':
    main()