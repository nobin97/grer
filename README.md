# emo_reco
Gradual Relaxation using Emotion Recognition

This python program provides movie suggestions based on your emotion using emotion-recognition.

Uses Keras library over tensorflow for emotion reco.

Movie suggestions are made in order to help improve the users mood based on his/her current emotion.

How it Works:-
  1)takes 5 images of the user through webcam with a 3 second delay in-between.
  2)performs emotion recogition on each image and outputs the emotion as result into an array.
  3)it then finds out the emotion with maximum count(or the Most Probable Emotion).
  4)Movie Genre is set according to the emotion (as shown in the code).
  5)Movie Title, Url and a short summary of the movie is displayed as a result.
  6)Repeats until user interrupts.
  
  
  
Movie API is provided by Yts.am
   https://yts.am/api
