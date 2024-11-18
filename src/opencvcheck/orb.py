import cv2

img  = cv2.imread('../dataset/nurmahal/nuramal.jpg',cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()

keypoints, descriptors = orb.detectAndCompute(img,None)

print("keypoints:",[kp.pt for kp in keypoints[:5]])
print("descriptors:",descriptors[:5])