# MobiGaze
This work is based on Google's ML kit Sample<br>
https://github.com/googlesamples/mlkit/tree/master/android/vision-quickstart <br>
Collaborators: <br>
<a href="https://github.com/oleeyoung520?tab=repositories">oleeyoung520</a> Email: 2015147520@yonsei.ac.kr <br>
### Before Cloning to Your Directory
This work requires <b>git-lfs</b> so you must install it first. (tflite file exceeds 100MB...)<br>
After installing git-lfs, on the Directory you want to clone this work, <br>
<pre><code>$git lfs install
Git LFS initialized.
$ git clone https://github.com/joonb14/MobiGaze.git
</code></pre>
### Summary
I mainly changed <b>FaceDetectorProcessor.java</b> and <b>FaceGraphic.java</b> <br>
Also deleted most of the source code that is not needed<br>
Added custom TensorFlow Lite model which is used for Gaze Estimation<br>
### Gaze Estimation Model
stored in asset folder. Created with Keras, converted to tflite.<br>
named "model.tflite"<br>
### It's Working but not Well
MobiGaze uses Personalized model. This example is based on my Data. So would not work well on any other people.<br>
Training Code will be uploaded soon.
