import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img  = cv.imread('../../dataset/Book Statue/WhatsApp Image 2024-11-25 at 19.01.18 (1).jpeg',cv.IMREAD_GRAYSCALE)

# check for FAST - opencv has FAST16 while we have used FAST9 though, so the number of keypoints would differ significantly 
# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()
# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
cv.imwrite('fast_true.png', img2)
# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
 
# check for BRIEF 
# Initiate FAST detector
star = cv.xfeatures2d.StarDetector_create()
# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
# find the keypoints with STAR
kp = star.detect(img,None)
# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)
print(des, "< - these are descriptors")
print( brief.descriptorSize() )
print( des.shape )


# ORB check! 

orb = cv.ORB_create()

keypoints, descriptors = orb.detectAndCompute(img,None)

print("keypoints:",[kp.pt for kp in keypoints[:5]])
print("descriptors:",descriptors[:5])