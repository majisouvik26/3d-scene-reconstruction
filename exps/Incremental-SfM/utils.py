import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def toHomogenous(pts):
    return np.vstack([pts[:, 0], pts[:, 1], np.ones(pts.shape[0])]).T.copy()


def toUnhomogenize(pts):
    return pts[:, :-1] / pts[:, None, -1]


def plot_points(img, points, path):
    img = img.copy()
    for point in points:
        x, y = point
        img = cv2.circle(img, (int(x), int(y)), radius=15, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(path, img)


def plot3d_points(points3d):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x, y, z = points3d[:, 0], points3d[:, 1], points3d[:, 2]
    ax.scatter(x, y, z, c=z, cmap='rainbow', s=1, alpha=1)
    plt.show()
    plt.close()


# Code snippet credits: Piazza
def plot3d_animation(points3d, tag, savepath):
    def rotate(angle):
        ax.view_init(azim=angle)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x, y, z = points3d[:, 0], points3d[:, 1], points3d[:, 2]
    ax.scatter(x, y, z, c=z, cmap='rainbow', s=1, alpha=1)

    if not os.path.exists(savepath):
        os.makedirs(savepath, exist_ok=True)
    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 361, 2), interval=50)
    rot_animation.save(f'{savepath}/rotation_{tag}.gif', dpi=80, writer='imagemagick')
