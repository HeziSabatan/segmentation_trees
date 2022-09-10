import cv2
import numpy as np


# show masked image:
def main():
    # define trackbars:
    def nothing(x):
        pass

    cv2.namedWindow('trackbars')
    cv2.createTrackbar('MIN_H', 'trackbars', 0, 179, nothing)
    cv2.createTrackbar('MIN_S', 'trackbars', 0, 255, nothing)
    cv2.createTrackbar('MIN_V', 'trackbars', 0, 255, nothing)
    cv2.createTrackbar('MAX_H', 'trackbars', 0, 179, nothing)
    cv2.createTrackbar('MAX_S', 'trackbars', 0, 255, nothing)
    cv2.createTrackbar('MAX_V', 'trackbars', 0, 255, nothing)
    cv2.createTrackbar('E_kernel', 'trackbars', 0, 15, nothing)
    cv2.createTrackbar('D_kernel', 'trackbars', 0, 15, nothing)

    # load the image:
    tree_image = cv2.imread('tree_image.jpg')
    # change to HSV:
    hsv = cv2.cvtColor(tree_image, cv2.COLOR_BGR2HSV)
    while True:

        # set trackbars:
        min_H = cv2.getTrackbarPos('MIN_H', 'trackbars')
        min_S = cv2.getTrackbarPos('MIN_S', 'trackbars')
        min_V = cv2.getTrackbarPos('MIN_V', 'trackbars')
        max_H = cv2.getTrackbarPos('MAX_H', 'trackbars')
        max_S = cv2.getTrackbarPos('MAX_S', 'trackbars')
        max_V = cv2.getTrackbarPos('MAX_V', 'trackbars')
        EROSION_KERNEL_SIZE = cv2.getTrackbarPos('E_kernel', 'trackbars')
        DILATION_KERNEL_SIZE = cv2.getTrackbarPos('D_kernel', 'trackbars')
        # create masked image:

        lower = np.array([min_H, min_S, min_V])
        upper = np.array([max_H, max_S, max_V])
        mask = cv2.inRange(hsv, lower, upper)

        # erode and dilate:
        erosion_kernel = np.ones((EROSION_KERNEL_SIZE, EROSION_KERNEL_SIZE), np.uint8)
        dilation_kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)
        erosion = cv2.erode(mask, erosion_kernel, 1)
        dilation = cv2.dilate(erosion, dilation_kernel, 1)

        # create 3D mask:
        mask_array = np.array((dilation, dilation, dilation)).transpose((1, 2, 0))
        masked_image = mask_array

        # create new image:
        changed_image = np.bitwise_and(tree_image, masked_image)
        # show image:
        cv2.imshow('trackbars', np.zeros((50, 750)))
        cv2.imshow('masked_image', changed_image)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            # define images to save:
            edge_image = cv2.Canny(dilation, 100, 200)
            tree_image_copy = tree_image.copy()
            tree_image_copy[edge_image > 0] = [0, 0, 255]
            concat_image = cv2.hconcat([tree_image_copy, changed_image])

            # save images:
            cv2.imwrite('mask_image.jpg', dilation)
            cv2.imwrite('new_image.jpg', changed_image)
            cv2.imwrite('concat_new_image.jpg', concat_image)
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
