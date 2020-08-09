# MobiGaze
This work is based on Google's ML kit<br>
https://github.com/googlesamples/mlkit/tree/master/android/vision-quickstart <br>
### Summary
I mainly changed <b>FaceDetectorProcessor.java</b> and <b>FaceGraphic.java</b> <br>
Also deleted most of the source code that is not needed<br>
Added custom TensorFlow Lite model which is used for Gaze Estimation<br>
### Gaze Estimation Model
stored in asset folder. Created with Keras, converted to tflite.<br>
named "model.tflite"<br>
