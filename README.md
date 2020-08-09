# MobiGaze
This work is based on Google's ML kit<br>
https://github.com/googlesamples/mlkit/tree/master/android/vision-quickstart <br>
### Before Cloning to Your Directory
This work requires <b>git-lfs</b> so you must install it first. (tflite file exceeds 100MB...)<br>
After installing git-lfs, on the Directory you want to clone this work, <br>
<pre><code>
$git lfs install
Git LFS initialized.
$ git clone https://github.com/joonb14/MobiGaze.git
</code></pre>
<br>
### Summary
I mainly changed <b>FaceDetectorProcessor.java</b> and <b>FaceGraphic.java</b> <br>
Also deleted most of the source code that is not needed<br>
Added custom TensorFlow Lite model which is used for Gaze Estimation<br>
### Gaze Estimation Model
stored in asset folder. Created with Keras, converted to tflite.<br>
named "model.tflite"<br>
