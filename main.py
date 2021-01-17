# This is a sample Python script.
import cv2
import numpy as np

# general config
win_name = 'Values'


def trackbarCreate():
    cv2.namedWindow(win_name)
    cv2.createTrackbar('Kernel Size', win_name, 3, 20, nothing)
    cv2.createTrackbar('dp', win_name, 100, 100, nothing)
    cv2.createTrackbar('min_dist', win_name, 10, 100, nothing)
    cv2.createTrackbar('param1', win_name, 30, 100, nothing)
    cv2.createTrackbar('param2', win_name, 20, 100, nothing)
    cv2.createTrackbar('min_rad', win_name, 20, 100, nothing)
    cv2.createTrackbar('max_rad', win_name, 100, 400, nothing)
    cv2.createTrackbar('change', win_name, 0, 1, passer)


def nothing(x):
    pass


def passer(x):
    main()


def method_kmeans(img):
    height, width, channel = img.shape
    # start preprocessing
    x = cv2.getTrackbarPos('Kernel Size', win_name)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (x, x))
    inImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    modImg = cv2.morphologyEx(inImg, cv2.MORPH_OPEN, kernel)
    modImg = cv2.morphologyEx(modImg, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('test', inImg)
    # cv2.imshow('test2', modImg)
    # cv2.imshow('test3', modImg2)
    # cv2.imshow('test4', nmImg)
    # end preprocessing
    flatImg = modImg.flatten()
    flatImg = np.float32(flatImg)
    x, labels, z = cv2.kmeans(flatImg, 3, None, (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 300, 1), 50,
                              cv2.KMEANS_RANDOM_CENTERS)
    flatLabels = labels.flatten()
    print(height)
    print(width)
    print(channel)
    labels2D = np.reshape(flatLabels, (height, width))
    print(len(labels2D[0]))
    notCells = np.zeros((height, width), np.uint8)
    aliveCells = np.zeros((height, width), np.uint8)
    cCells = np.zeros((height, width), np.uint8)

    for x in range(height):
        for y in range(width):
            value = labels2D[x, y]
            if value == 0:
                notCells[x, y] = modImg[x, y]
            elif value == 1:
                aliveCells[x, y] = modImg[x, y]
            elif value == 2:
                cCells[x, y] = modImg[x, y]
    dp = cv2.getTrackbarPos('dp', win_name) / 100
    min_dist = cv2.getTrackbarPos('min_dist', win_name)
    par1 = cv2.getTrackbarPos('param1', win_name)
    par2 = cv2.getTrackbarPos('param2', win_name)
    min_rad = cv2.getTrackbarPos('min_rad', win_name)
    max_rad = cv2.getTrackbarPos('max_rad', win_name)

    cv2.imshow('notCells', notCells)
    cv2.imshow('aliveCells', aliveCells)
    cv2.imshow('cCells', cCells)
    cv2.imshow('modImg', modImg)
    cv2.imshow('img', img)
    # cv2.imshow('gray', gray)
    return notCells, aliveCells, cCells


def blob_detection(img, drawImg=None):
    params = cv2.SimpleBlobDetector_Params()
    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 100
    # Set Circularity filtering parameters
    params.filterByCircularity = False
    params.minCircularity = 0.001
    # Set Convexity filtering parameters
    params.filterByConvexity = False
    params.minConvexity = 0.001
    # Set inertia filtering parameters
    params.filterByInertia = False
    params.minInertiaRatio = 0.001
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs
    keypoints = detector.detect(img)
    print(f"Number of keypoints are {len(keypoints)}")
    blank = np.zeros((1, 1))
    if drawImg is None:
        blobs = cv2.drawKeypoints(img, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        blobs = cv2.drawKeypoints(drawImg, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return blobs


def const_ench(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab", lab)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # cv2.imshow('l_channel', l)
    # cv2.imshow('a_channel', a)
    # cv2.imshow('b_channel', b)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # cv2.imshow('CLAHE output', cl)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))
    # cv2.imshow('limg', limg)

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def detectCircles(count_img, draw_img=None):
    cell_count = 0
    circles = cv2.HoughCircles(count_img, cv2.HOUGH_GRADIENT, 1, 2, param1=25, param2=20, minRadius=2, maxRadius=20)
    if circles is not None:
        detected_circles = np.uint16(np.around(circles))
        for (x, y, r) in detected_circles[0, :]:
            if draw_img is not None:
                cv2.circle(draw_img, (x, y), r, (0, 255, 0), 1)
            cell_count += 1
    print("Cell number is : {}".format(cell_count))
    return draw_img, cell_count


def method_hsv(img):
    # TODO Add guided image filtering
    # mod_img = cv2.GaussianBlur(img, (1, 1), 0)  # TODO Here will come the guided image filtering
    kernel = np.ones((5, 5), np.float32) / 25
    mod_img = cv2.filter2D(img, -1, kernel)
    # cv2.imshow('blur', mod_img)  # Show blurred image
    ench_img = const_ench(mod_img)  # Brightness Adjustment
    # cv2.imshow('ench', ench_img)  # Show enhanced image
    ench_img = cv2.cvtColor(ench_img, cv2.COLOR_BGR2HSV)  # Turn enhanced image to HSV for processing
    hue, saturation, value = cv2.split(ench_img)
    dst = cv2.adaptiveThreshold(saturation, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)
    # ret, dst = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Threshold image using Otsu
    masked = cv2.bitwise_and(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), dst)  # Apply Otsu threshold to image
    x = 2  # Kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (x, x))  # TODO Get kernel from trackbars
    modImg = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel)  # Use opening to eliminate small noises
    modImg = cv2.dilate(modImg, kernel)  # Use dilate to eliminate small noises

    testImg = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel)  # Use opening to eliminate small noises
    testImg = cv2.morphologyEx(testImg, cv2.MORPH_CLOSE, kernel)

    # resultImg, cell_amount = detectCircles(modImg, img)
    resultImg = blob_detection(modImg, img)
    # cv2.imshow('grayscale', cv2.cvtColor(ench_img, cv2.COLOR_BGR2GRAY))
    # cv2.imshow('threshold', dst)
    cv2.imshow('masked', masked)
    cv2.imshow('modImg', modImg)
    cv2.imshow('testImg', testImg)
    cv2.imshow('resultImg', resultImg)  # Show the final image


def main():
    img = cv2.imread('cropped.jpg')
    # cv2.imshow('img', img)
    method_hsv(img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # TODO create an argument that decides whether or not use the kmeans and create trackbar according to that
    # trackbarCreate()
    main()
