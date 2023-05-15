import cv2 as cv


def vid_convert(filename):

    cap = cv.VideoCapture(filename)
    count = 0
    # if not cap.isOpened():
    #     exit(0)
    while True:

        retval, frame = cap.read()
        if frame is None:
            break
        cv.imwrite("C:\\Users\\SACHIN\\motion_training\\frames\\frame%d.jpg" % count, frame)
        count += 1


if __name__ == '__main__':
    for i in range(23):
        vid_convert("C:\\Users\\SACHIN\\motion_training\\videos\\" + str(i) + ".mp4")
