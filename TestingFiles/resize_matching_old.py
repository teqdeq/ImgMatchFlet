import cv2
import os
import pickle
import numpy as np
from osc4py3.as_eventloop import *
from osc4py3 import oscbuildparse
from pathlib import Path
import time
# Start the OSC system.
osc_startup()

# Make client channels to send packets.
osc_udp_client("127.0.0.1", 8000, "tester")


# Define path
pathImages = 'ImagesQuery'
pathMovies = 'Movies'

# small thresh hold mean small feature set
orb = cv2.KAZE_create(threshold=0.0007)

# Set the threshold of minimum Features detected to give a positive, around 20 to 30
thres = 30

# List Images and Print out their Names and how many there are in the Folder
images = []
classNamesImages = []
myListImages = os.listdir(pathImages)

print('Total Image Detected', len(myListImages))
# this will read in the images
for cl in myListImages:
    imgCur = cv2.imread(f'{pathImages}/{cl}', 0)
    if imgCur is None:
        continue
    images.append(imgCur)
    # delete the file extension
    classNamesImages.append(os.path.splitext(cl)[0])



# List Movies and Print out their Names and how many there are in the Folder
Movies = []
myListMovies = os.listdir(pathMovies)

print('Total Movies Detected', len(myListMovies))


def saveFeature(images_features, images_name):
    print("saving feature for next time use")
    with open('features.pkl', 'wb') as f:
        pickle.dump([images_features, images_name], f)

def is_need_to_computer_feature(myListImages):
    is_file_exist = os.path.exists('features.pkl')
    if is_file_exist:
        print("checking for new images in folder")
        f = open('features.pkl', 'rb')
        images_features, images_name = pickle.load(f)
        should_computer_feature = not np.array_equal(images_name,myListImages)
        # return the feature with indocator to check either image has been changed in folder
        return (should_computer_feature, images_features)
    else:
        print("No pkl file found")
        return (True, [])


# this will find the matching points in the images
def findDes(images):
    # First check either we have already computer features or not
    desList = []

    (should_computer_feature, images_features) = is_need_to_computer_feature(myListImages)

    if should_computer_feature:
        print("Start computing features")    
        for img in images:
            # resile image berfore fetching feature
            img = cv2.resize(img, (320, 320))
            kp, des = orb.detectAndCompute(img, None)
            desList.append(des)
        # Save these feature for next time with Images name list
        saveFeature(desList, myListImages)
    else:
        desList = images_features

    return desList

# this will compare the matches and find the corresponding image
def findID(img, desList):
    image_matching_time = time.time()

    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                # the distance of the matches in comparison to each other
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    # uncomment to see how many positive matches, according to this the thres is set
    # print(matchList)

    if len(matchList) != 0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
    print("matching time --- %s seconds ---" % (time.time() - image_matching_time))
    return finalVal



print("start finding feature")
desList = findDes(images)
print(len(desList))

# open Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)


while True:
    success, img2 = cap.read()
    imgOriginal = img2.copy()
    
    if img2 is None:
        continue
    # convert Camera to Grayscale
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(img2, (320, 320))

    # if Matching with Image in List, send the respective Name
    id = findID(img2, desList)

    if id != -1:
        # put text for the found Image
        cv2.putText(imgOriginal, classNamesImages[id], (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

        #cap = cv2.VideoCapture('Movies/Banksy.mp4')

        # while True:
        #     success, img = cap.read()
        #     cv2.imshow('Video', img)
        #     cv2.waitKey(0)


        msg = oscbuildparse.OSCMessage("/test/me", ",s", [classNamesImages[id]])
        osc_send(msg, "tester")
        osc_process()

# # if not Matching with any Image in List, send 'none'
    if -1 == id:
#
#         capNone = cv2.VideoCapture('Movies/Black.mp4')
#
#         while True:
#             success, img3 = capNone.read()
#             cv2.imshow('Video', img3)
#             cv2.waitKey(1)

        msg = oscbuildparse.OSCMessage("/test/me", ",s", ["none"])
        osc_send(msg, "tester")
        osc_process()


    # show the final Image
    cv2.imshow('img2', imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
