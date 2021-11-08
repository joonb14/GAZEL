# GAZEL
This work is official implementation of GAZEL framework which is published in <a href="http://www.percom.org/">PerCom 2021(GAZEL: Runtime Gaze Tracking for Smartphones)</a> .
![overview](https://user-images.githubusercontent.com/30307587/109145148-71f31f80-77a5-11eb-94a7-34a47a1f19d5.png)
This work is heavily based on Google's Firebase ML kit Sample(2020, June Version)<br>
https://github.com/googlesamples/mlkit/tree/master/android/vision-quickstart <br>
inspired by: <a href="https://gazecapture.csail.mit.edu/">Eye Tracking for Everyone(2016 CVPR)</a><br><br>
Collaborators: <br>
<a href="https://github.com/oleeyoung520?tab=repositories">oleeyoung520</a> Email: 2015147520@yonsei.ac.kr <br>
<a href="https://github.com/Yeeun55">Yeeun55</a> Email: joyce9559@naver.com <br>
<a href="https://github.com/yeokyeong46">yeokyeong46</a> Email: yeokyeong46@gmail.com <br>

## Sample Videos
This work is based on Galaxy Tab S6. <br>
We trained model with data collected using  <a href="https://github.com/joonb14/MLKitGazeDataCollectingButton.git"> MLKitGazeDataCollectingButton</a><br>
We also provide a method to utilize our Tablet model to Smartphones by calibration.<br>
Red dot represent "Raw Gaze Estimation Output"<br>
Blue dot represent "Moving Averaged Output" (if Calibration is done, moving average on calibrated output, else on raw output)<br>
Green dot represent "Calibrated Raw Output"<br>
Trained with Galaxy Tab S6 data ,tested on Galaxy Tab S6<br>
[![TabS6](https://img.youtube.com/vi/rhO5kjTn0Ts/0.jpg)](https://www.youtube.com/watch?v=rhO5kjTn0Ts)<br>
Trained with Galaxy Tab S6 data, tested and calibrated on Galaxy S9+<br>
[![S9+](https://img.youtube.com/vi/iwPjHlGeRpw/0.jpg)](https://youtu.be/iwPjHlGeRpw)<br>
### Summary
I mainly changed <b>FaceDetectorProcessor.java, LivePreviewActivity.java</b> and <b>FaceGraphic.java</b> <br>
Also deleted most of the source code that is not needed<br>
Added a guide to load and use custom TensorFlow Lite model which is used for Gaze Estimation<br>

## Details
#### Gaze Estimation Model
<img src="https://user-images.githubusercontent.com/30307587/109145286-a6ff7200-77a5-11eb-86ff-41925981af10.png" alt="model" style="width:800px;"/>

Model should be stored in asset folder. I recommend to create model with Keras, then converted it to the TFlite model.<br>
You can check the output also on Logcat. <b>TAG is "MOBED_GazePoint"</b><br>
<img src="https://user-images.githubusercontent.com/30307587/109152350-b8994780-77ae-11eb-882b-0ef26e723d86.png" alt="logcat"/>

#### Before we start...
GAZEL uses <b>Personalized model</b>. This model is heavily dependent on my facial appearances (Wearing glasses & in Lab environment). So would not work well on other person.(Have already tested...)<br>
So, I decided to exclude the .tflite model and provide training source codes and data collecting application for you to follow.<br>
The Keras model training & TensorFlow Lite conversion Code is provided in <a href="https://github.com/joonb14/MLKitGazeDataCollectingButton.git"> MLKitGazeDataCollectingButton</a>.<br>

#### For the Calibration
We used  5 points calibration with translation, and rescaling.<br>
5 points are TopLeft, TopRight, BottomLeft, BottomRight, and Center<br>
We also tried to provide SVR calibration. However, multi output SVR doesn't exist in android. So we are using 2 regressors(with android <a href="https://github.com/yctung/AndroidLibSVM">libsvm</a>) for each x and y coordinate (The calibration experiments on the paper are conducted on the "server" with scikit-learn, not on the "smartphones"). This does not work as well as the linear calibration, so we recommend to use linear calibration.<br>


## TFLite Configuration<a id="tflite_config"></a>
If you want to use custom TFLite model with our GAZEL Framework. First check  configuration options below(in <b>FaceDetectorProcessor.java</b> ). We provide Face bitmap, Left/Right Eye Grids, Face Grid.
We used 1-channel bitmap for enhancing gaze estimation accuracy, but like other papers which use 3-channel RGB images as input, we provide 3-channel image mode. You can change the mode with THREE-CHANNEL flag. We also provide various options for you to test your model with various model inputs, so try to create gaze estimation model with various inputs!

```java
private final boolean USE_EULER = true; // true: use euler x,y,z as input
private final boolean USE_FACE = false; // true: use face x,y,z as input
private final boolean USE_EYEGRID = false; // true: use eye_grid as input
private final boolean USE_FACEGRID = true; // true: use face_grid as input
private final boolean THREE_CHANNEL = false; // false for Black and White image, true for RGB image
private final boolean calibration_mode_SVR = false; // false for translation & rescale. true for SVR
private final boolean CORNER_CALIBRATION = false; // false for translation & rescale with center, true for only 4 corners
```
Above configuration flags are about  switching modes, now below configuration values are specific values for initializing modes.
```java
private final double SACCADE_THRESHOLD = 300; // distance for classifying FIXATION and SACCADE
private final int resolution = 64; // for eye and face
private final int grid_size = 50; // for eye_grids
private final int face_grid_size = 25; // for face_grid
private final int FPS = 30; // for calibration count
private final int SKIP_FRAME = 10; // for calibration count
private final int COST = 40; // for SVR
private final int GAMMA = 1; // for SVR
private final int QUEUE_SIZE = 20; // for moving average
private final float EYE_OPEN_PROB = 0.0f; //empirical value
```

In case you put your TFLite model in the <b>"GAZEL/GazeTracker/app/src/main/assets/custom_models/eval/"</b> directory, you must change the below line in <b>LivePreviewActivity.java</b>, change

```java
InputStream inputStream = getAssets().open("custom_models/eval/[your_model_name]].tflite");
```

then follow the [issues](#issues)
#### Issues
TensorFlow Lite Conversion. Before you load your tflite model, you must check the input details to make sure input order is correct.<br>
In case you are using python interpreter,

```python
import tensorflow as tf
tflite = tf.lite.Interpreter(model_path="path/to/model.tflite")
tflite.get_input_details()
```
example output will be

```
[{'name': 'left_eye',
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
```
Then reorder your inputs in <b>FaceDetectorProcessor.java</b> <a id="issues"></a>
```java
inputs = new float[][][][][]{left_4d, right_4d, euler, facepos, face_grid}; // make sure the order is correct
```
## Custom Device Configuration
This work is based on Tablet devices. So if you want to use this framework on Smartphones, you need to follow some instructions.<br>

* First, you need Tablet device for training base Gaze Estimation CNN Model.
* Second, you need to collect <b>"Ground Truth Gaze Data"</b>  with <a href="https://github.com/joonb14/MLKitGazeDataCollectingButton.git"> MLKitGazeDataCollectingButton</a>.
* Third, you need to train your Gaze Estimation CNN Model with <a href="https://github.com/joonb14/MLKitGazeDataCollectingButton/blob/master/GAZEL.ipynb">provided python code<a/>.
* Fourth, you need to follow [TFLite Configuration](#tflite_config)
* Fifth, follow the instructions below
Change the configuration options below(in <b>FaceDetectorProcessor.java</b> ) with your Target device spec.
```java
private final boolean isCustomDevice = true;
//custom device
private final float customDeviceWidthPixel = 1440.0f;
private final float customDeviceWidthCm = 7.0f;
private final float customDeviceHeightPixel = 2960.0f;
private final float customDeviceHeightCm = 13.8f;
private final float customDeviceCameraXPos = 4.8f; // in cm | at Android coordinate system where use top left corner as (0,0)
private final float customDeviceCameraYPos = -0.3f; // in cm | at Android coordinate system where use top left corner as (0,0)
//original device
private final float originalDeviceWidthPixel = 1600.0f;
private final float originalDeviceWidthCm = 14.2f;
private final float originalDeviceHeightPixel = 2560.0f;
private final float originalDeviceHeightCm = 22.5f;
private final float originalDeviceCameraXPos = 7.1f; // in cm | at Android coordinate system where use top left corner as (0,0)
private final float originalDeviceCameraYPos = -0.5f; // in cm | at Android coordinate system where use top left corner as (0,0)
```

set the <b>isCustomDevice</b> flag to true, then change all of the <b>customDevice[option]</b> values

* Lastly, run the GAZEL application, and you must click <b>"START CALIB"</b> button to start calibration and use it as Gaze Tracker.

#### Tips
Collect data as much as you can before training your model. Recommend you to use different head position with different light conditions. 

## Discussion
It is well known knowledge to use massive image dataset to train feature extraction layers then only train fully connected layers for targeted environment. I as a mobile system developer however wanted to try using mobile embedded sensors to improve gaze estimation accuracy. That's why I tried to build up new application using collected sensor outputs as well as front camera frames. (And that's why I didn't use the massive image dataset for training).
Also I believe our work is the first open source smartphone gaze tracking framework. I hope this little proof of concept framework would help your research. Thank you.
