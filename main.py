# This is a sample Python script.
import cv2
import numpy as np

#def initialize():
#    global
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    cv2.imshow('blur', blur)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    cv2.imshow('adaptive', adaptive)
    ret, dst = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masked = cv2.bitwise_and(gray, dst)
    return masked

def main():
    img = cv2.imread('yeastcount.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = preprocess(img)
    #detector = cv2.SimpleBlobDetector_create()
    #keypoints = detector.detect(gray)
    #blank = np.zeros((1, 1))
    #img_key = cv2.drawKeypoints(gray, keypoints, blank, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('original', gray)
    cv2.imshow('tresh', mask)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    #initialize()
    main()





