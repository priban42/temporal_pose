import numpy as np
import cv2

def get_depth_undistorter(K, shape):
    inv_K = np.linalg.inv(K)
    undistorter = np.zeros(shape)
    for y, x in np.ndindex(shape):
        pix_vect = np.array([x, y, 1])
        world_vect = inv_K@pix_vect
        world_vect = world_vect/np.linalg.norm(world_vect)
        world_vect_proj = np.dot(world_vect, np.array([0, 0, 1]).T)
        undistorter[y, x] = np.linalg.norm(world_vect_proj)
    return undistorter

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(new_img[y, x]/10000, "m")

if __name__ == "__main__":
    K = np.array([[614.193, 0, 326.268],
                    [0, 614.193, 238.851],
                    [0, 0, 1]])
    img = cv2.imread('D:\\BOP_datasets\\ObjectTrackingDataset\\test\\000001\\depth\\' + f'{0:06}' + ".png",cv2.IMREAD_UNCHANGED)
    undistorter = get_depth_undistorter(K, np.shape(img))
    PATH = 'D:\\BOP_datasets\\ObjectTrackingDataset\\test\\000006\\depth\\'
    for x in range(80):
        file_name = 'D:\\BOP_datasets\\ObjectTrackingDataset\\test\\000006\\depth\\' + f'{x:06}' + ".png"
        img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        new_img = (undistorter*img).astype('uint16')
        cv2.imwrite(file_name, new_img)
        print(x)

    # new_img = undistort_depth_image(img, K)
    # cv2.imshow('image', new_img)
    # cv2.setMouseCallback('image', mouse_click)
    # cv2.waitKey(0)

cv2.destroyAllWindows()

