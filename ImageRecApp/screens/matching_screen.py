import cv2
import time
import os
import pickle
import numpy as np
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock

class MatchingScreen(Screen):
    def __init__(self, **kwargs):
        super(MatchingScreen, self).__init__(**kwargs)

        # Define path
        pathImages = 'ImagesQuery'
        pathMovies = 'MoviesQuery'

        # Small threshold means small feature set
        orb = cv2.KAZE_create(threshold=0.0007)

        # Set the threshold of minimum features detected to give a positive, around 20 to 30
        thres = 30

        # List Images and Print out their Names and how many there are in the Folder
        images = []
        classNamesImages = []
        myListImages = os.listdir(pathImages)

        print('Total Images Detected', len(myListImages))
        # This will read in the images
        for cl in myListImages:
            imgCur = cv2.imread(f'{pathImages}/{cl}', 0)
            if imgCur is None:
                continue
            images.append(imgCur)
            # Delete the file extension
            classNamesImages.append(os.path.splitext(cl)[0])

        def saveFeature(images_features, images_name):
            print("Saving features for next time use")
            with open('features.pkl', 'wb') as f:
                pickle.dump([images_features, images_name], f)

        def is_need_to_compute_feature(myListImages):
            is_file_exist = os.path.exists('features.pkl')
            if is_file_exist:
                print("Checking for new images in folder")
                f = open('features.pkl', 'rb')
                images_features, images_name = pickle.load(f)
                should_compute_feature = not np.array_equal(images_name, myListImages)
                # Return the feature with an indicator to check whether images have changed in the folder
                return should_compute_feature, images_features
            else:
                print("No pkl file found")
                return True, []

        # This will find the matching points in the images
        def findDes(images):
            # First check whether we have already computed features or not
            desList = []

            should_compute_feature, images_features = is_need_to_compute_feature(myListImages)

            if should_compute_feature:
                print("Start computing features")
                for img in images:
                    # Resize image before fetching features
                    img = cv2.resize(img, (320, 320))
                    kp, des = orb.detectAndCompute(img, None)
                    desList.append(des)
                # Save these features for next time with the images name list
                saveFeature(desList, myListImages)
            else:
                desList = images_features

            return desList

        # This will compare the matches and find the corresponding image
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
                        # The distance of the matches in comparison to each other
                        if m.distance < 0.75 * n.distance:
                            good.append([m])
                    matchList.append(len(good))
            except:
                pass

            if len(matchList) != 0:
                if max(matchList) > thres:
                    finalVal = matchList.index(max(matchList))
            print("Matching time --- %s seconds ---" % (time.time() - image_matching_time))
            return finalVal

        # Track the number of frames since a valid match
        match_count = 0

        def process_frame(dt):
            nonlocal match_count
            success, img2 = cap.read()
            imgOriginal = img2.copy()

            if img2 is None:
                return

            # Convert Camera to Grayscale
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            img2 = cv2.resize(img2, (320, 320))

            # If matching with an image in the list, transition to the ChooseScreen
            id = findID(img2, desList)
            if id != -1:
                match_count += 1
                if match_count >= 5:
                    self.manager.current = "choose"
            else:
                match_count = 0

            # Show the final image
            cv2.imshow('img2', imgOriginal)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                Clock.unschedule(process_frame)
                cap.release()
                cv2.destroyAllWindows()

        # Start finding features
        print("Start finding features")
        desList = findDes(images)
        print(len(desList))

        # Open Webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

        # Schedule the image processing at a fixed interval (e.g., 30 frames per second)
        Clock.schedule_interval(process_frame, 1 / 30)  # 30 frames per second
