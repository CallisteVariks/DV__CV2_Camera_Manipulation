import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

# Capture video using the webcam
cap = cv2.VideoCapture(0) 

while(True):
    ret, frame = cap.read()
    img = cv2.flip(frame, 1)
    img = cv2.flip(img, 0)
    cv2.imshow('Video Capture', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# Set the codec for the video file and save it accordingly
fileName = 'output.avi'
imgSize = (640,480)
frame_per_second = 30.0
writer = cv2.VideoWriter(fileName, cv2.VideoWriter_fourcc(*'MJPG'), frame_per_second, imgSize)

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        writer.write(frame)
        cv2.imshow('Video Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()


# Load the created file and play the video accordingly
fileName = 'output.avi'

cap = cv2.VideoCapture(fileName)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()


# Change the size of the window
scaling_factorx=0.5
scaling_factory=0.5

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factorx, fy=scaling_factory, interpolation=cv2.INTER_AREA)
    img = frame
    cv2.imshow('Smaller Window', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Change color of the frames
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Enhance the contrast
def equalizeHistColor(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    img = equalizeHistColor(frame)
    cv2.imshow('Histogram Equalization', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Warp the image
def WarpImage(frame):
    ax, bx = 10.0, 100
    ay, by = 20.0, 120
    img = np.zeros(frame.shape, dtype=frame.dtype)
    rows, cols = img.shape[:2]
    for i in range(rows):
        for j in range(cols):
            offset_x = int(ax*math.sin(2*math.pi*i/bx))
            offset_y = int(ay*math.cos(2*math.pi*j/by))
            if i+offset_y < rows and j+offset_x < cols:
                img[i, j] = frame[(i + offset_y) % rows, (j+offset_x) %cols]
            else:
                img[i, j]=0
    return img

def equalizeHistColor(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    img[:, :, 2] = cv2.equalizeHist(img[:,:,2])
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    if ret == 1:
        img = equalizeHistColor(WarpImage(frame))
    else:
        img = equalizeHistColor(frame)

    cv2.imshow('Warped', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Create an aura near the head as an optical flow
cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    prvs = next
    cv2.imshow('Optical Flow', bgr)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# Transform the captured image using the edge detection technique
img = cv2.imread('/assets/20160505_142020.jpg', 0)
edges = cv2.Canny(img, 20, 60, 1)
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()



# Other edge detection techniques such as Laplacian and Sobel
kernelSize = 11 
# Parameters
parameter1 = 20
parameter2 = 60
intApertureSize = 1

# Laplacian
cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    frame = cv2.Laplacian(frame, cv2.CV_64F)
    cv2.imshow('Laplacian', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
cap.release()
cv2.destroyAllWindows()

# Canny
cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    frame = cv2.Canny(frame,parameter1,parameter2,intApertureSize)
    cv2.imshow('Canny', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#Sobel - X
cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    frame = cv2.Sobel(frame,cv2.CV_64F,1,0, ksize=kernelSize)
    cv2.imshow('Sobel - X',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# Sobel - Y
cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    frame = cv2.Sobel(frame,cv2.CV_64F,0,1, ksize=kernelSize) 
    cv2.imshow('Sobel - Y',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# Gaussian blur
cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame, (kernelSize,kernelSize), 0, 0)
    cv2.imshow('Gaussian',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# Median blur
cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    frame = cv2.medianBlur(frame, kernelSize)
    cv2.imshow('Median Blur',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# Blur
cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    frame = cv2.blur(frame,(kernelSize,kernelSize))
    cv2.imshow('Blur',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



# Use edge detection as a mask and superimpose the image with cv2.bitwise_and() function
kernelSize = 21
# Edge Detection Parameter
parameter1 = 10
parameter2 = 40
intApertureSize = 1

cap = cv2.VideoCapture(0)
while (True):
    ret, frame1 = cap.read()
    frame = cv2.GaussianBlur(frame1, (kernelSize, kernelSize), 0, 0)
    edge = cv2.Canny(frame, parameter1, parameter2, intApertureSize)
    frame = cv2.bitwise_and(frame1, frame1, mask=edge)
    # Display results
    cv2.imshow('Super Impose', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
