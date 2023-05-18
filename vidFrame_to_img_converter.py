import cv2 as cv


def vid_convert(filename, count_in):
    cap = cv.VideoCapture(filename)
    if not cap.isOpened():
        exit(0)
    count = count_in
    while True:
        retval, frame = cap.read()
        if frame is None:
            break
        cv.imwrite("D:\\CBT\\bagZipping\\frames\\frame%d.jpg" % count, frame)
        count += 1
    return count


if __name__ == '__main__':
    count = 0
    for i in range(23):
        count = vid_convert("D:\\CBT\\bagZipping\\" + str(i) + ".mp4", count)

if __name__ == '__main__':
    for i in range(23):
        vid_convert("C:\\Users\\SACHIN\\motion_training\\videos\\" + str(i) + ".mp4")
