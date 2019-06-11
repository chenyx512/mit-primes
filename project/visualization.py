import cv2

video = cv2.VideoCapture('data/camera_front.avi')
# 1208, 1920, 3
cnt = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (240, 180))


while video.isOpened():
    ret, frame = video.read()
    if ret:
        cnt += 1
    else:
        break

    frame = cv2.resize(frame, (240, 180))
    out.write(frame)

    if cnt % 1000 == 0:
        break

video.release()
out.release()
cv2.destroyAllWindows()
print(cnt)