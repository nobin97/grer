import cv2
import time  
import os
import sys
import warnings
import requests
import random
import numpy as np
try:
    print("\nSetting things up. Please wait...")
    print("\nSpecial Thanks to YTS YIFY Movies Torrents website for providing with its Movies API.\n")
    premature_exit = 0
    skip_netlookup = 0
    imshow = 0
    use_burst_mode = 0   #(1 for yes, 0 for no)
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from keras.models import load_model
    from utils.datasets import get_labels
    from utils.inference import apply_offsets
    from utils.preprocessor import preprocess_input

    #initialisations...
    emotion_model_path = './models/emotion_model.hdf5'
    cam_id = 0
    rounds = 1
    emotion_labels = get_labels('fer2013')
    emotion_offsets = (20, 40)
    emotion_array = []
    old_emotion = []
    
    #loading model and face-cascade...
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model(emotion_model_path)
    emotion_target_size = emotion_classifier.input_shape[1:3]

    input("\nReady. Press Enter to Start...")
    try:
        if os.path.exists('./Subject_images') == False:
            print("\nCreating Subject Image directory...", end = "  ")
            os.mkdir('./Subject_images')
            print("Done.")
        if os.path.exists('./img_debug') == False:
            print("\nCreating Debug Image directory...", end = "  ")
            os.mkdir('./img_debug')
            print("Done.")
    except:
        print("\nCouldn't create directory. Make sure all necessary permissions are given to the Application.")
        input("Press Enter to Exit...")
        premature_exit = 1
        sys.exit()
            
    #Pinging Google to check for network connection...
    if skip_netlookup == False:
        while True:
            Url = "https://www.google.com"
            print("\nLooking for a Network Connection...", end = "  ")
            try:
                response = requests.get(url = Url)
                print("Network Connected.")
                break
            except:
                print("No Network Connection.")
                ch = input("Check Again?(y/n)")
                if ch == "y" or ch == "Y":
                    pass
                elif ch == "n" or ch == "N":
                    print("\nExiting...")
                    premature_exit = 1
                    sys.exit()
                else:
                    pass

    ch = input("\nShow capture window?(y/n)")
    if ch == "n" or ch == "N":
        pass
    else:
        imshow = 1
    ch = input("\nUse Burst mode?(y/n)")
    if ch == "n" or ch == "N":
        pass
    else:
        use_burst_mode = 1
            
    
    #Checking if Camera is available...
    while True:
        print("\nInitializing Camera...", end = "  ")
        cap = cv2.VideoCapture(cam_id)
        if cap is None or not cap.isOpened():
            print("\n\nCouldn't connect with Camera. Make sure you have a working webcam!!")
            ch = input("Check Again?(y/n)")
            if ch == "y" or ch == "Y":
                pass
            elif ch == "n" or ch == "N":
                print("\nExiting...")
                premature_exit = 1
                sys.exit()
            else:
                pass
        elif cap.read() == (False, None):
            print("\n\nCamera looks Busy. Close any other application that is using camera and Try again!!")
            ch = input("Check Again?(y/n)")
            if ch == "y" or ch == "Y":
                pass
            elif ch == "n" or ch == "N":
                print("\nExiting...")
                premature_exit = 1
                sys.exit()
            else:
                pass          
        else:
            print("Done.")
            break
        
    if use_burst_mode == 1:
        print("\n*Using Burst Mode*")
        
    #capturing image and detecting emotion one by one...
    while True:
        cnt = 0
        cam_img_init = cap.read()
        time.sleep(2)
        dir_list = []
        emotion_array = []
        print("\nRound:",rounds, "| Capturing images...", sep =" ")
        rounds += 1
        while(cnt < 5):
            ret, frame = cap.read()
            if os.path.exists('./Subject_images') == False:
                os.mkdir('./Subject_images')
            path = "./Subject_images/subject_img.jpg"
            cv2.imwrite(path, frame)    
            print(cnt + 1, ". Captured ", path, sep = "")
            cnt += 1
            
            try:
                bgr_image = cv2.imread(path)
                gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

                faces = face_cascade.detectMultiScale(gray_image, scaleFactor = 1.1, minNeighbors = 5,
                                        minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
                for (a, b, c, d) in faces:
                    cv2.rectangle(bgr_image, (a, b), (a+c, b+d), (0, 255, 255), 4)
                if imshow == 1:
                    cv2.moveWindow('test', 40,30)
                    cv2.imshow('test', bgr_image)
                    cv2.waitKey(2)
                    
                debug_dir_path = "./img_debug/debug_img_" + str(cnt - 1) + ".jpg"
                cv2.imwrite(debug_dir_path, bgr_image)
                        
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
                    emotion_label_arg = np.argmax(emotion_prediction)
                    emotion_text = emotion_labels[emotion_label_arg]
                try:
                    print("Deduced Emotion:", emotion_text, end = "\n\n")
                    emotion_array.append(emotion_text)
                    del emotion_text
                    if use_burst_mode != 1:
                        time.sleep(3)
                except:
                    print(" *No face detected*\n")
                    if use_burst_mode != 1:
                        time.sleep(3)
                    continue
               # cv2.destroyAllWindows()
            except:
                continue
                
        if emotion_array == []:
            print("\nCouldn't detect any face!! Try adjusting the camera angle / Remove the camera lid...")
            cap.release()
            input("Press Enter to continue...")
            print("\nRestarting...")
            cap = cv2.VideoCapture(cam_id)
            continue
        elif len(emotion_array) < 3:
            print("Emotion count too low to deduce the most probable emotion...\nRestarting...")
            continue
        
        #Set genre & Fetch functions...
        def fetch_movie(movie_genre):
            max_retry = 0
            movie_found = 0
            Url = "https://yts.am/api/v2/list_movies.json"
            Params = {"genre" : movie_genre, "limit" : 5}
            print("\nFetching Movie List... Genre = ", movie_genre, end = "\n")
            
            #Retry 10 times...
            while max_retry < 10:
                try:
                    response = requests.get(url = Url, params = Params)
                    json_data = response.json()
                    movie_arr = json_data["data"]["movies"]
                    movie_found = 1
                    return movie_arr
                    break
                except:
                    max_retry += 1
                    continue
            if movie_found == 0:
                return -1
                
        def set_genre(emotion_with_maxcnt):
            if ("angry" in emotion_with_maxcnt
                or "fear" in emotion_with_maxcnt
                or "sad" in emotion_with_maxcnt
                or "disgust" in emotion_with_maxcnt
                or "surprise" in emotion_with_maxcnt):
                movie_genre = ["Comedy","Fantasy","Romance","Film-Noir","Animation"][random.randint(0,4)]
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
        print("Most Probable Emotion:",emotion_with_maxcnt)
        if emotion_with_maxcnt == old_emotion:
            print("\n*No change in Emotion*")
            continue
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
                for i in response_movies:
                    print(id,") Movie Name: ", "\"", i["title"], "\"", "\n", "\t  Url: ", i["url"], "\nShort Summary: ", i["summary"], "\n", sep="")
                    id += 1
                    if cam_release == 1:
                        cap = cv2.VideoCapture(cam_id)                            
            else:
                print("\nCouldn't fetch list. Check Internet/VPN Settings or Try again later!!")
                cap.release()
                cam_release = 1
                ch = str(input("Retry?(y/n)"))
                if ch == "y" or ch == "Y":
                    pass
                elif ch == "n" or ch == "N":
                    print("\nExiting...")
                    premature_exit = 1
                    sys.exit()
                else:
                    pass
except:
    if premature_exit != 1:
        try:
            cap.release()
            cv2.destroyAllWindows()
            print("\nExecution Stopped by User/SysInterrupt!!")
        except:
            cv2.destroyAllWindows()
            print("\nExecution Stopped by User/SysInterrupt!!")
