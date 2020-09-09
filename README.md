# MobiGaze
This work is based on Google's ML kit Sample(2020, June Version)<br>
https://github.com/googlesamples/mlkit/tree/master/android/vision-quickstart <br>
inspired by: <a href="https://gazecapture.csail.mit.edu/">Eye Tracking for Everyone(2016 CVPR)</a><br><br>
Collaborators: <br>
<a href="https://github.com/oleeyoung520?tab=repositories">oleeyoung520</a> Email: 2015147520@yonsei.ac.kr <br>
<a href="https://github.com/Yeeun55">Yeeun55</a> Email: joyce9559@naver.com <br>
<a href="https://github.com/yeokyeong46">yeokyeong46</a> Email: yeokyeong46@gmail.com <br><br>

### Summary
I mainly changed <b>FaceDetectorProcessor.java, LivePreviewActivity.java</b> and <b>FaceGraphic.java</b> <br>
Also deleted most of the source code that is not needed<br>
Added custom TensorFlow Lite model which is used for Gaze Estimation<br>
### Gaze Estimation Model
stored in asset folder. Created with Keras, converted to tflite.<br>
named <b>"checkpoint/facepos.tflite"</b> model has best accuracy<br>
You can check the output also on Logcat. <b>TAG is "MOBED_GazePoint"</b><br>
### Working on...
MobiGaze uses Personalized model. This example is based on my Data(Wearing glasses). So would not work well on other person.<br>
Now working on Calibration. Typically we are going to use 5 points calibration with translation, and rescaling.<br>
5 points are TopLeft, TopRight, BottomLeft, BottomRight, and Center<br>
We also tried to provide SVR calibration. However, multi output SVR doesn't exist in android. So we are using 2 regressors(with android <a href="https://github.com/yctung/AndroidLibSVM">libsvm</a>) for x and y coordinate.<br>
However the problem is... I cannot get the right cost and gamma for SVR... and it seems to need much more calibration point than 5. <br>
So we use default calibration method with translation, and rescaling.<br>
Keras model training & conversion Code will be uploaded soon.<br><br>
### Issues
TensorFlow Lite Conversion. Before you load your tflite model, you must check the input details to make sure input order is correct.<br>
In case you are using python interpreter,
<pre><code>import tensorflow as tf

tflite = tf.lite.Interpreter(model_path="path/to/model.tflite")
tflite.get_input_details()
</code></pre>
example output will be
<pre><code>[{'name': 'left_eye',
  'index': 4,
  'shape': array([ 1, 64, 64,  1], dtype=int32),
  'dtype': numpy.float32,
  'quantization': (0.0, 0)},
 {'name': 'right_eye',
  'index': 56,
  'shape': array([ 1, 64, 64,  1], dtype=int32),
  'dtype': numpy.float32,
  'quantization': (0.0, 0)},
 {'name': 'euler',
  'index': 1,
  'shape': array([1, 1, 1, 3], dtype=int32),
  'dtype': numpy.float32,
  'quantization': (0.0, 0)},
 {'name': 'facepos',
  'index': 3,
  'shape': array([1, 1, 1, 2], dtype=int32),
  'dtype': numpy.float32,
  'quantization': (0.0, 0)},
 {'name': 'face_grid',
  'index': 2,
  'shape': array([ 1, 25, 25,  1], dtype=int32),
  'dtype': numpy.float32,
  'quantization': (0.0, 0)}]
</code></pre>
Then reorder your inputs in <b>FaceDetectorProcessor.java</b>
<pre><code>inputs = new float[][][][][]{left_4d, right_4d, euler, facepos, face_grid}; // make sure the order is correct
</code></pre>
