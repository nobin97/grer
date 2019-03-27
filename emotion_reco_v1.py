import cv2
import time  
import os
import requests
import random
import numpy as np
try:
    print("\nSetting things up. Please wait...\n")
    from keras.models import load_model
    from statistics import mode
    from utils.datasets import get_labels
    from utils.inference import detect_faces
    from utils.inference import draw_text
    from utils.inference import draw_bounding_box
    from utils.inference import apply_offsets
    from utils.inference import load_detection_model
    from utils.preprocessor import preprocess_input

    #initialisations...
    emotion_model_path = './models/emotion_model.hdf5'
    emotion_labels = get_labels('fer2013')
    emotion_offsets = (20, 40)
    emotion_array = []
    old_emotion = []
    emotion_prev = 0

    #loading models and cascade...
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model(emotion_model_path)

    #emoreco_start = input("\nReady. Press Enter to Start...\n")
    input("\nReady. Press Enter to Start...")
    print("\nInitializing Camera...")
    cap = cv2.VideoCapture(0)
    print("Done.")
    
    #capturing images...
    while True:
        cnt=0
        cam_img_init = cap.read()
        time.sleep(2)
        print("\nCapturing images...")
        while(cnt < 5):
            ret,frame = cap.read()
            path = "./Subject_images/subject_img_" + str(cnt) + ".jpg"
            cv2.imwrite(path, frame)
            print("Captured", path)
            cnt += 1
            time.sleep(3)
        print("\nDone. Deducing Emotions...")

        #loading captured images and deducing emotion...
        while True:    
            try:
                emotion_target_size = emotion_classifier.input_shape[1:3]
                dir_list = []
                emotion_array = []
                temp_arr = os.listdir("./Subject_images")
                for dir_item in temp_arr:
                    if ".jpg" in dir_item or ".jpeg" in dir_item or ".png" in dir_item:
                        dir_list.append(dir_item)
                        
                for subject_image_path in dir_list:
                    imagePath = "./Subject_images/" + subject_image_path
                    bgr_image = cv2.imread(imagePath)
                    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
                    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

                    faces = face_cascade.detectMultiScale(gray_image, scaleFactor = 1.1, minNeighbors = 5,
                                            minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

                    for face_coordinates in faces:
                        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                        gray_face = gray_image[y1:y2, x1:x2]
                        try:
                            gray_face = cv2.resize(gray_face, (emotion_target_size))
                        except:
                            continue

                        gray_face = preprocess_input(gray_face, True)
                        gray_face = np.expand_dims(gray_face, 0)
                        gray_face = np.expand_dims(gray_face, -1)
                        emotion_prediction = emotion_classifier.predict(gray_face)
                        #emotion_probability = np.max(emotion_prediction)
                        emotion_label_arg = np.argmax(emotion_prediction)
                        emotion_text = emotion_labels[emotion_label_arg]
                    try:
                        emotion_array.append(emotion_text)
                    except:
                        continue
            except:
                print("\nOops! An Unknown Exception Occured. Resuming...")
                cap = cv2.VideoCapture(0)
                break
            if emotion_array == []:
                print("\nCouldn't detect any face!! Try adjusting the camera angle / Remove the camera lid...")
                cap.release()
                input("Press Enter to continue...")
                print("\nRestarting...")
                cap = cv2.VideoCapture(0)
                break
            print("\n",emotion_array, sep="")
            
            #Set genre & Fetch functions...
            def fetch_movie(movie_genre):
                Url = "https://yts.am/api/v2/list_movies.json"
                Params = {"genre" : movie_genre, "limit" : 5}
                print("\nFetching Movie List... Genre = ", movie_genre, end = "\n")
                try:
                    response = requests.get(url = Url, params = Params)
                    json_data = response.json()
                    movie_arr = json_data["data"]["movies"]
                    return movie_arr
                except:
                    return -1
                
            def set_genre(emotion_with_maxcnt):
                if ("angry" in emotion_with_maxcnt
                    or "fear" in emotion_with_maxcnt
                    or "sad" in emotion_with_maxcnt
                    or "disgust" in emotion_with_maxcnt
                    or "surprise" in emotion_with_maxcnt):
                    movie_genre = ["Comedy","Fantasy","Romance","Film-Noir"][random.randint(0,3)]
                else:
                    movie_genre = ["Drama","Fantasy","Sci-Fi","Romance",
                               "Action","Thriller","Mystery",
                               "Animation","Adventure"][random.randint(0,8)]
                return movie_genre        

            #Getting Emotion with max count (Most probable emotion)
            set_arr = set(emotion_array)
            emotion_dict = {}
            for emotion in set_arr:
                cnt = 0
                for emotion1 in emotion_array:
                    if emotion1 == emotion:
                        cnt += 1
                emotion_dict[emotion] = cnt

            max_cnt = max(emotion_dict.values())

            emotion_with_maxcnt = []
            for emotion, count in emotion_dict.items():
                if count == max_cnt:
                    emotion_with_maxcnt.append(emotion)
            print("\nMost Probable Emotion:",emotion_with_maxcnt)
            if emotion_with_maxcnt == old_emotion and emotion_prev == 0:
                print("\n*No change in Emotion*")
                break
            else:
                old_emotion = emotion_with_maxcnt
                

            movie_genre = set_genre(emotion_with_maxcnt)
            getmovie = 0
            cam_release = 0
            while(getmovie != 1):
                response_movies = fetch_movie(movie_genre)
                if response_movies != -1:
                    id = 1
                    getmovie = 1
                    emotion_prev = 0
                    for i in response_movies:
                        print(id,") Movie Name: ", "\"", i["title"], "\"", "\n", "\t  Url: ", i["url"], "\nShort Summary: ", i["summary"], "\n", sep="")
                        id += 1
                        if cam_release == 1:
                            cap = cv2.VideoCapture(0)                            
                else:
                    print("\nCouldn't fetch list. Check Internet/VPN Settings or Try again later!!")
                    cap.release()
                    cam_release = 1
                    emotion_prev = 1
                    ch = str(input("Retry?(y/n)"))
                    if ch == "y" or ch == "Y":
                        pass
                    else:
                        print("\nRestarting...")
                        cap = cv2.VideoCapture(0)
                        getmovie = 1
            break
except:
    try:
        cap.release()
        print("\nKeyboardInterrupt. Stopped by User!!")
    except:
        print("\nKeyboardInterrupt. Stopped by User!!")
        
