#Library/package
!pip install opencv-python

#package -  image segment,analyse work
import cv2

# helps to display the image
import matplotlib.pyplot as plt

# reading image as input
image= cv2.imread('rose.jpg') # reading image as input

#If image is not uploaded properly
if image is None:
    raise ValueError("Image not loaded. Check the file path.")

# Display the size of the image
print(image.shape)

plt.imshow(image)
plt.show()

# BGR - blue, green, and red format, RGB - Red Green Blue format
image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#imageshow
plt.imshow(image_rgb)
#or give off
plt.axis('off')
plt.title('RGB Image')
plt.show()

# GRAY SCALE CONVERSION (Convert the picture into the gray scale format)
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image,cmap='gray')
plt.axis('off')
plt.title('Grayscale Image')
plt.show()

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Print grayscale image properties
print("Grayscale image shape: ")
print(gray_image.shape)
print("Grayscale image dtype: ")
print(gray_image.dtype)
print(f"Grayscale image size: ")
print(gray_image.size)


# Check if grayscale image is empty
if gray_image is None or gray_image.size == 0:
    raise ValueError("Grayscale image is empty.")

# Check depth
if gray_image.dtype != 'uint8':
    raise ValueError("Grayscale image has incorrect depth. It should be CV_8U.")

width, height = 640, 480
resized_image = cv2.resize(image, (width, height))

plt.imshow(resized_image)

#Blurring (Smoothing)
blurred_image=cv2.GaussianBlur(image,(5,5),0) #5x5 kernal
edges= cv2.Canny(gray_image,threshold1=100, threshold2=200)
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.title('Blurred Image')
plt.show()

plt.imshow(blurred_image)

#Corner Detection**
from skimage.feature import hog
from skimage import color
# Convert image to grayscale
gray_image = color.rgb2gray(image_rgb)
# Extract HOG features
features, hog_image = hog(gray_image, visualize=True, feature_vector=True)

# Display HOG image
plt.imshow(hog_image, cmap='gray')
plt.axis('off')
plt.show()

#SIFT (Scale-Invariant Feature Transform)

sift = cv2.SIFT_create()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(gray_image, None)

# Check if keypoints are detected
if keypoints is None:
    raise ValueError("No keypoints detected. Check the input image and parameters.")

image_with_keypoints = cv2.drawKeypoints(image_rgb, keypoints, None)
plt.imshow(image_with_keypoints)
plt.axis('off')
plt.show()

#ORB (Oriented FAST and Rotated BRIEF)
orb = cv2.ORB_create() #creates descriptors and keypoints
# Detect keypoints and descriptors
keypoints, descriptors = orb.detectAndCompute(gray_image, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image_rgb, keypoints, None)
plt.imshow(image_with_keypoints)
plt.axis('off')
plt.show()