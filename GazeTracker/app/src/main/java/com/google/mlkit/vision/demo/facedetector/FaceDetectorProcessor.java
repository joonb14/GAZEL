/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.facedetector;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.PointF;
import androidx.renderscript.Allocation;
import androidx.renderscript.RenderScript;

import android.graphics.Rect;
import android.os.Environment;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.RelativeLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.LivePreviewActivity;
import com.google.mlkit.vision.demo.Queue;
import com.google.mlkit.vision.demo.R;
import com.google.mlkit.vision.demo.VisionProcessorBase;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceContour;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceLandmark;

//TF Lite
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

//LibSvm
import umich.cse.yctung.androidlibsvm.LibSVM;

/**
 * MobiGaze: Personalized Mobile Gaze Tracker
 * Based on Galaxy Tab S6
 */
public class FaceDetectorProcessor extends VisionProcessorBase<List<Face>> {
    private final boolean isGAZEL = true;
    /**
     * Cross-Phone Implementation
     * tested with Galaxy S9+
     * */
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
    /**
     * Config Modes
     * */
    private final boolean USE_EULER = true; // true: use euler x,y,z as input
    private final boolean USE_FACE = false; // true: use face x,y,z as input
    private final boolean USE_EYEGRID = false; // true: use eye_grid as input
    private final boolean USE_FACEGRID = false; // true: use face_grid as input
    private final boolean THREE_CHANNEL = false; // false for Black and White image, true for RGB image
    private final boolean calibration_mode_SVR = false; // false for translation & rescale. true for SVR
    private final boolean CORNER_CALIBRATION = false; // false for translation & rescale with center, true for only 4 corners
    /**
     * Config Values
     * */
    private final float CANVAS_WIDTH = 1348.0f; // for eye and face
    private final float CANVAS_HEIGHT = 2398.0f; // for eye and face
    private final double SACCADE_THRESHOLD = 300; // distance for classifying FIXATION and SACCADE
    private final int resolution = 64; // for eye and face
    private final int grid_size = 50; // for eye_grids
    private final int face_grid_size = 25; // for face_grid
    private final int FPS = 30; // for calibration count
    private final int SKIP_FRAME = 10; // for calibration count
    private final int COST = 40; // for SVR
    private final float GAMMA = 0.3f; // for SVR
    private final int QUEUE_SIZE = 20; // for moving average
    private final float EYE_OPEN_PROB = 0.0f; //empirical value
    /**
     * Constant Values
     */
    private static final float EYE_BOX_RATIO = 1.4f;
    private static final String TAG = "MOBED_FaceDetector";

    public static Interpreter tflite;
    public static Bitmap image;
    private final FaceDetector detector;
    public float leftEyeleft, leftEyetop, leftEyeright, leftEyebottom;
    public float rightEyeleft, rightEyetop, rightEyeright, rightEyebottom;
    private RenderScript RS;
    private ScriptC_singlesource script;

    private float[][][][] left_4d, right_4d, face_grid, lefteye_grid, righteye_grid ,face_input, euler, facepos, right_eye_right_top, left_eye_left_bottom, right_eye_left_bottom, left_eye_right_top;
    //private float[][][] face_input;
    float[][][][][] inputs;

    private float yhatX =0, yhatY=0;
    LibSVM svmX;
    LibSVM svmY;
    Button calibration_button;
    private boolean calibration_flag=false;
    Context Fcontext;
    private int calibration_phase = 0;
    private boolean calibration_model_exist = false;
    private String basedir;

    private boolean moving_average_start = true;
    private float moving_average_X, moving_average_Y;

    private float  top_left_mean_X, top_left_mean_Y;
    private float top_right_mean_X, top_right_mean_Y;
    private float bottom_left_mean_X, bottom_left_mean_Y;
    private float bottom_right_mean_X, bottom_right_mean_Y;
    private float center_mean_X, center_mean_Y;

    private float center_offset_X, center_offset_Y;
    //CORNER CALIBRATION
    private float top_left_offset_X, top_left_offset_Y;
    private float bottom_right_offset_X, bottom_right_offset_Y;
    //TRANSLATION & RESCALE CALIBRATION
    private float tlxscale, tlyscale, trxscale, tryscale, blxscale, blyscale, brxscale, bryscale;

    private float calib_X, calib_Y;
    private boolean calib_moving_average = true;
    private float calib_moving_average_X, calib_moving_average_Y;
    //Bitmap returns mirrored image so needs mirroring matrix at first
    private Matrix matrix;

    //Saccade detection
    private boolean isSaccade = false;

    public FaceDetectorProcessor(Context context, FaceDetectorOptions options ) {
        super(context);
        this.Fcontext = context;
        calibration_model_exist = false;
        calibration_button = (Button)((Activity)context).findViewById(R.id.calibration);
        Log.v(MANUAL_TESTING_LOG, "Face detector options: " + options);
        detector = FaceDetection.getClient(options);
        RS = RenderScript.create(context);
        script = new ScriptC_singlesource(RS);
        svmX = new LibSVM();
        svmY = new LibSVM();
        basedir = Environment.getExternalStorageDirectory().getPath()+"/MobiGaze/";
        float[] mirrorY = {
                -1, 0, 0,
                0, 1, 0,
                0, 0, 1
        };
        matrix = new Matrix();
        matrix.setValues(mirrorY);
        calibration_button.setOnClickListener(new Button.OnClickListener() {
            @Override
            public void onClick(View view) {
                // TODO : click event
                if(!calibration_flag){
                    Log.d(TAG,"Calibration Start");
                    calibration_flag =true;
                    calibration_model_exist = false;
                    calibration_button.setText(R.string.calibration_end);
                    File logFile1 = new File(basedir+"trainX.txt");
                    File logFile2 = new File(basedir+"trainY.txt");
                    File svmXfile = new File(basedir+"svmX.model");
                    File svmYfile = new File(basedir+"svmY.model");
                    if (logFile1.exists()) {
                        try {
                            logFile1.delete();
                            logFile2.delete();
                            svmXfile.delete();
                            svmYfile.delete();
                        }
                        catch (Exception e) {
                            // TODO Auto-generated catch block
                            e.printStackTrace();
                        }
                    }
                }
            }
        });
    }

    @Override
    public void stop() {
        super.stop();
        detector.close();
    }

    @Override
    protected Task<List<Face>> detectInImage(InputImage image) {
        return detector.process(image);
    }

    @Override
    protected void onSuccess(@NonNull List<Face> faces, @NonNull GraphicOverlay graphicOverlay) {
        /**
         * TODO
         * MOBED
         * Notice!
         * Real "Left eye" would be "Right eye" in the face detection. Because camera is left and right reversed.
         * And all terms in face detection would follow direction of camera preview image
         * */
        Log.d("LATENCY","Face Detected");
        for (Face face : faces) {
            if (face != faces.get(0)) break;
            //MOBED
            //This is how you get coordinates, and crop left and right eye
            //Look at https://firebase.google.com/docs/ml-kit/detect-faces#example_2_face_contour_detection for details.
            //We specifically used Eye Contour's point 0 and 8.
            if (face.getRightEyeOpenProbability() != null && face.getLeftEyeOpenProbability() != null) {
                float rightEyeOpenProb = face.getRightEyeOpenProbability();
                float leftEyeOpenProb = face.getLeftEyeOpenProbability();
                Log.d(TAG, "Right Eye open: "+ rightEyeOpenProb+", Left Eye open: "+leftEyeOpenProb);
                if(rightEyeOpenProb<EYE_OPEN_PROB /*|| leftEyeOpenProb <EYE_OPEN_PROB*/) continue; // in my case my left eye is too small for the use
            }
            else {
                Log.d(TAG, "Eye open prob is null");
            }
            try {
                List<PointF> leftEyeContour = face.getContour(FaceContour.LEFT_EYE).getPoints();
                List<PointF> rightEyeContour = face.getContour(FaceContour.RIGHT_EYE).getPoints();
                float righteye_leftx = rightEyeContour.get(0).x;
                float righteye_lefty = rightEyeContour.get(0).y;
                float righteye_rightx = rightEyeContour.get(8).x;
                float righteye_righty = rightEyeContour.get(8).y;
                float lefteye_leftx = leftEyeContour.get(0).x;
                float lefteye_lefty = leftEyeContour.get(0).y;
                float lefteye_rightx = leftEyeContour.get(8).x;
                float lefteye_righty = leftEyeContour.get(8).y;
                float righteye_centerx = (righteye_leftx + righteye_rightx)/2.0f;
                float righteye_centery = (righteye_lefty + righteye_righty)/2.0f;
                float lefteye_centerx = (lefteye_leftx + lefteye_rightx)/2.0f;
                float lefteye_centery = (lefteye_lefty + lefteye_righty)/2.0f;
                float lefteyeboxsize = (lefteye_centerx-lefteye_leftx)*EYE_BOX_RATIO;
                float righteyeboxsize = (righteye_centerx-righteye_leftx)*EYE_BOX_RATIO;
                leftEyeleft = lefteye_centerx - lefteyeboxsize;
                leftEyetop = lefteye_centery + lefteyeboxsize;
                leftEyeright = lefteye_centerx + lefteyeboxsize;
                leftEyebottom = lefteye_centery - lefteyeboxsize;
                rightEyeleft = righteye_centerx - righteyeboxsize;
                rightEyetop = righteye_centery + righteyeboxsize;
                rightEyeright = righteye_centerx + righteyeboxsize;
                rightEyebottom = righteye_centery - righteyeboxsize;
                Bitmap leftBitmap=Bitmap.createBitmap(image, (int)leftEyeleft,(int)leftEyebottom,(int)(lefteyeboxsize*2), (int)(lefteyeboxsize*2), matrix, false);
                Bitmap rightBitmap=Bitmap.createBitmap(image, (int)rightEyeleft,(int)rightEyebottom,(int)(righteyeboxsize*2), (int)(righteyeboxsize*2), matrix, false);
                if (leftBitmap.getHeight() > resolution){
                    leftBitmap = Bitmap.createScaledBitmap(leftBitmap, resolution,resolution,false);
                }
                if (rightBitmap.getHeight() > resolution){
                    rightBitmap = Bitmap.createScaledBitmap(rightBitmap, resolution,resolution,false);
                }
                //Renderscript converts RGBA value to YUV's Y value.
                //After RenderScript, Y value will be stored in Red pixel value
                if (!THREE_CHANNEL) {
                    Allocation inputAllocation = Allocation.createFromBitmap(RS, leftBitmap);
                    Allocation outputAllocation = Allocation.createTyped(RS, inputAllocation.getType());
                    script.invoke_process(inputAllocation, outputAllocation);
                    outputAllocation.copyTo(leftBitmap);
                    inputAllocation = Allocation.createFromBitmap(RS, rightBitmap);
                    outputAllocation = Allocation.createTyped(RS, inputAllocation.getType());
                    script.invoke_process(inputAllocation, outputAllocation);
                    outputAllocation.copyTo(rightBitmap);
                }
                if (leftBitmap.getHeight() < resolution){
                    leftBitmap = Bitmap.createScaledBitmap(leftBitmap, resolution,resolution,false);
                }
                if (rightBitmap.getHeight() < resolution){
                    rightBitmap = Bitmap.createScaledBitmap(rightBitmap, resolution,resolution,false);
                }
                int[] eye = new int[resolution*resolution];
                /**
                 * Euler
                 * EulerX, EulerY, EulerZ
                 * */
                if(USE_EULER) {
                    euler = new float[1][1][1][3];
                    euler[0][0][0][0] = face.getHeadEulerAngleX();
                    euler[0][0][0][1] = face.getHeadEulerAngleY();
                    euler[0][0][0][2] = face.getHeadEulerAngleZ();
                }
                /**
                 * Left Eye
                 * */
                leftBitmap.getPixels(eye,0,resolution,0,0,resolution,resolution);
                if(!THREE_CHANNEL) {
                    left_4d = new float[1][resolution][resolution][1];
                }
                else left_4d = new float[1][resolution][resolution][3];
                for(int y = 0; y < resolution; y++) {
                    for (int x = 0; x < resolution; x++) {
                        int index = y * resolution + x;
                        left_4d[0][y][x][0] = ((eye[index] & 0x00FF0000) >> 16)/(float)255;
                        if(THREE_CHANNEL) {
                            left_4d[0][y][x][1] = ((eye[index] & 0x0000FF00) >> 8)/(float)255;;
                            left_4d[0][y][x][2] = (eye[index] & 0x000000FF)/(float)255;;
                        }
                    }
                }
                /**
                 * Right Eye
                 * */
                rightBitmap.getPixels(eye,0,resolution,0,0,resolution,resolution);
                if(!THREE_CHANNEL) {
                    right_4d = new float[1][resolution][resolution][1];
                }
                else right_4d = new float[1][resolution][resolution][3];
                for(int y = 0; y < resolution; y++) {
                    for (int x = 0; x < resolution; x++) {
                        int index = y * resolution + x;
                        right_4d[0][y][x][0] = ((eye[index] & 0x00FF0000) >> 16)/(float)255;
                        if(THREE_CHANNEL) {
                            right_4d[0][y][x][1] = ((eye[index] & 0x0000FF00) >> 8)/(float)255;;
                            right_4d[0][y][x][2] = (eye[index] & 0x000000FF)/(float)255;;
                        }
                    }
                }
                /**
                 * Left, Right Eye Contour for SAGE
                 * */

                if (graphicOverlay.isImageFlipped) {
                    leftEyeleft = graphicOverlay.getWidth() - (leftEyeleft * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset);
                    leftEyeright = graphicOverlay.getWidth() - (leftEyeright * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset);
                    rightEyeleft = graphicOverlay.getWidth() - (rightEyeleft * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset);
                    rightEyeright = graphicOverlay.getWidth() - (rightEyeright * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset);
                } else {
                    leftEyeleft = leftEyeleft * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset;
                    leftEyeright = leftEyeright * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset;
                    rightEyeleft = rightEyeleft * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset;
                    rightEyeright = rightEyeright * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset;
                }
                rightEyebottom = rightEyebottom * graphicOverlay.scaleFactor - graphicOverlay.postScaleHeightOffset;
                rightEyetop = rightEyetop * graphicOverlay.scaleFactor - graphicOverlay.postScaleHeightOffset;
                leftEyebottom = leftEyebottom * graphicOverlay.scaleFactor - graphicOverlay.postScaleHeightOffset;
                leftEyetop = leftEyetop * graphicOverlay.scaleFactor - graphicOverlay.postScaleHeightOffset;
                right_eye_right_top = new float[1][1][1][2];
                right_eye_right_top[0][0][0][0] = rightEyeright/CANVAS_WIDTH;
                right_eye_right_top[0][0][0][1] = rightEyetop/CANVAS_HEIGHT;
                left_eye_left_bottom = new float[1][1][1][2];
                left_eye_left_bottom[0][0][0][0] = leftEyeleft/CANVAS_WIDTH;
                left_eye_left_bottom[0][0][0][1] = leftEyebottom/CANVAS_HEIGHT;
                right_eye_left_bottom = new float[1][1][1][2];
                right_eye_left_bottom[0][0][0][0] = rightEyeleft/CANVAS_WIDTH;
                right_eye_left_bottom[0][0][0][1] = rightEyebottom/CANVAS_HEIGHT;
                left_eye_right_top = new float[1][1][1][2];
                left_eye_right_top[0][0][0][0] = leftEyeright/CANVAS_WIDTH;
                left_eye_right_top[0][0][0][1] = leftEyetop/CANVAS_HEIGHT;
                if(isGAZEL) {
                    right_eye_right_top[0][0][0][1] = right_eye_right_top[0][0][0][1] - right_eye_left_bottom[0][0][0][1];
                    right_eye_right_top[0][0][0][0] = right_eye_left_bottom[0][0][0][0] - right_eye_right_top[0][0][0][0];
                    left_eye_right_top[0][0][0][1] = left_eye_right_top[0][0][0][1] - left_eye_left_bottom[0][0][0][1];
                    left_eye_right_top[0][0][0][0] = left_eye_left_bottom[0][0][0][0] - left_eye_right_top[0][0][0][0];
                    Log.d("VALUE_CHECK", "right_eye_right_top: "+ right_eye_right_top[0][0][0][0] +","+right_eye_right_top[0][0][0][1]);
                    Log.d("VALUE_CHECK", "left_eye_right_top: "+ left_eye_right_top[0][0][0][0] +","+left_eye_right_top[0][0][0][1]);
                }
                /**
                 * Face
                 * */
                //int resolution = 224;
                Rect facePos = face.getBoundingBox();
                float faceboxWsize = facePos.right - facePos.left;
                float faceboxHsize = facePos.bottom - facePos.top;
                if(USE_FACE) {
                    Bitmap faceBitmap=Bitmap.createBitmap(image, (int)facePos.left,(int)facePos.top,(int)faceboxWsize, (int)faceboxHsize, matrix, false);
                    faceBitmap = Bitmap.createScaledBitmap(faceBitmap, resolution,resolution,false);
                    int[] face_pix = new int[resolution*resolution];
                    faceBitmap.getPixels(face_pix,0,resolution,0,0,resolution,resolution);
                    if(!THREE_CHANNEL) {
                        face_input = new float[1][resolution][resolution][1];
                    }
                    else face_input = new float[1][resolution][resolution][3];
                    for(int y = 0; y < resolution; y++) {
                        for (int x = 0; x < resolution; x++) {
                            int index = y * resolution + x;
                            face_input[0][y][x][0] = ((face_pix[index] & 0x00FF0000) >> 16)/(float)255.0f;
                            if(THREE_CHANNEL) {
                                face_input[0][y][x][1] = ((face_pix[index] & 0x0000FF00) >> 8)/(float)255.0f;;
                                face_input[0][y][x][2] = (face_pix[index] & 0x000000FF)/(float)255.0f;;
                            }
                        }
                    }
                }
                /**
                 * Facepos
                 * */
                float faceCenterX = (facePos.right + facePos.left)/2.0f;
                float faceCenterY = (facePos.bottom + facePos.top)/2.0f;
                float face_X, face_Y;
                if (graphicOverlay.isImageFlipped) {
                    face_X = graphicOverlay.getWidth() - (faceCenterX * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset);
                } else {
                    face_X = faceCenterX * graphicOverlay.scaleFactor - graphicOverlay.postScaleWidthOffset;
                }
                face_Y = faceCenterY * graphicOverlay.scaleFactor - graphicOverlay.postScaleHeightOffset;
                facepos = new float[1][1][1][2];
                facepos[0][0][0][0] = face_X;
                facepos[0][0][0][1] = face_Y;
                /**
                 * Face Grid
                 * */
                int image_width = image.getWidth();
                int image_height = image.getHeight();
                float w_start,h_start,w_num,h_num;
                if(USE_FACEGRID) {
                    //left, bottom, width, height
                    w_start = Math.round(face_grid_size * (facePos.left / (float) image_width));
                    h_start = Math.round(face_grid_size * (facePos.top / (float) image_height));
                    w_num = Math.round(face_grid_size * ((faceboxWsize) / (float) image_width));
                    h_num = Math.round(face_grid_size * ((faceboxHsize) / (float) image_height));

                    face_grid = new float[1][face_grid_size][face_grid_size][1];
                    for (int h = 0; h < face_grid_size; h++) {
                        for (int w = 0; w < face_grid_size; w++) {
                            if (w >= w_start && w <= w_start + w_num && h >= h_start && h <= h_start + h_num) {
                                face_grid[0][h][(face_grid_size - 1) - w][0] = 1;
                            } else face_grid[0][h][(face_grid_size - 1) - w][0] = 0;
                        }
                    }
                }
                /**
                 * Eye Grids
                 * Use to use these as inputs, but recognized just using eyes results in better results
                 * */
                if(USE_EYEGRID) {
                    image_width = image.getWidth();
                    image_height = image.getHeight();
                    //left, bottom, width, height
                    w_start = Math.round(grid_size*(leftEyeleft/(float)image_width));
                    h_start = Math.round(grid_size*(leftEyebottom/(float)image_height));
                    w_num = Math.round(grid_size*((2*lefteyeboxsize)/(float)image_width));
                    h_num = Math.round(grid_size*((2*lefteyeboxsize)/(float)image_height));

                    lefteye_grid = new float[1][grid_size][grid_size][1];
                    for(int h=0; h<grid_size; h++){
                        for(int w=0; w<grid_size; w++) {
                            if (w>=w_start && w<=w_start+w_num && h>=h_start && h<=h_start+h_num){
                                lefteye_grid[0][h][(grid_size-1)-w][0] = 1;
                            }
                            else lefteye_grid[0][h][(grid_size-1)-w][0] = 0;
                        }
                    }

                    w_start = Math.round(grid_size*(rightEyeleft/(float)image_width));
                    h_start = Math.round(grid_size*(rightEyebottom/(float)image_height));
                    w_num = Math.round(grid_size*((2*righteyeboxsize)/(float)image_width));
                    h_num = Math.round(grid_size*((2*righteyeboxsize)/(float)image_height));

                    righteye_grid = new float[1][grid_size][grid_size][1];
                    for(int h=0; h<grid_size; h++){
                        for(int w=0; w<grid_size; w++) {
                            if (w>=w_start && w<=w_start+w_num && h>=h_start && h<=h_start+h_num){
                                righteye_grid[0][h][(grid_size-1)-w][0] = 1;
                            }
                            else righteye_grid[0][h][(grid_size-1)-w][0] = 0;
                        }
                    }
                }
                /**
                 * Wrap them up to use them as input for TensorFlow Lite model
                 * */
                Log.d("LATENCY","Input Parced");
                if(USE_EULER) {
                    //gazel_shared_ver9
                    inputs = new float[][][][][]{facepos, right_4d, left_eye_right_top, euler, right_eye_right_top, left_4d};
                }
                else {
                    inputs = new float[][][][][]{left_4d, right_4d};
                }

                // To use multiple input and multiple output you must use the Interpreter.runForMultipleInputsOutputs()
                float[][] output = new float[1][2];
                Map<Integer, Object> outputs = new HashMap<>();
                outputs.put(0, output);
                Log.d("LATENCY","Input Wrapped");
                try {
                    // Clear out Files for Gaze Estimation
                    if(calibration_model_exist){
                        File testXfile = new File(basedir+"X.txt");
                        File testYfile = new File(basedir+"Y.txt");
                        File outXfile = new File(basedir+"outX.txt");
                        File outYfile = new File(basedir+"outY.txt");
                        if (testXfile.exists()) {
                            testXfile.delete();
                            testYfile.delete();
                            outXfile.delete();
                            outYfile.delete();
                        }
                    }
                    //Run TFLite
                    tflite.runForMultipleInputsOutputs(inputs, outputs);
                    //tflite.run(face_input, output);
                    //The output x,y will be stored to below variables
                    yhatX = output[0][0];
                    yhatY = output[0][1];
                    Log.d("LATENCY","Gaze Predicted");
                    if(isCustomDevice) {
                        //TODO
                        Log.d("CROSSDEVICE","yhatX,yhatY: (" + yhatX+","+yhatY+")");
                        float a = yhatX - (originalDeviceWidthPixel/originalDeviceWidthCm)*originalDeviceCameraXPos;
                        float b = yhatY - (originalDeviceHeightPixel/originalDeviceHeightCm)*originalDeviceCameraYPos;
                        Log.d("CROSSDEVICE","a,b: (" + a+","+b+")");
                        float aprime = a * (originalDeviceWidthCm/originalDeviceWidthPixel);
                        float bprime = b * (originalDeviceHeightCm/originalDeviceHeightPixel);
                        Log.d("CROSSDEVICE","aprime,bprime: (" + aprime+","+bprime+")");
                        float resultX = (aprime + customDeviceCameraXPos)*customDeviceWidthPixel/customDeviceWidthCm;
                        float resultY = (bprime + customDeviceCameraYPos)*customDeviceHeightPixel/customDeviceHeightCm;
                        Log.d("CROSSDEVICE","resultX,resultY: (" + resultX+","+resultY+")");
                        yhatX = resultX;
                        yhatY = resultY;
                    }

                    /**
                     * MOBED Moving Average Implementation
                     * Using Queue
                     */
                    if(moving_average_start){
                        moving_average_X = yhatX;
                        moving_average_Y = yhatY;
                        moving_average_start = false;
                    }
                    else {
                        double distance = Math.sqrt(Math.pow(moving_average_X-yhatX,2) + Math.pow(moving_average_Y-yhatY,2));
                        if(distance > SACCADE_THRESHOLD) {
                            isSaccade = true;
                            Log.d("MOBED_SACCADE","SACCADE, distance: "+distance + " | yhat: "+yhatX+","+yhatY+ " | moving_average: "+moving_average_X+","+moving_average_Y);
                        }
                        else {
                            isSaccade = false;
                            Log.d("MOBED_SACCADE","FIXATION, distance: "+distance + " | yhat: "+yhatX+","+yhatY+ " | moving_average: "+moving_average_X+","+moving_average_Y);
                        }

                        moving_average_X = moving_average_X*0.6f+yhatX*0.4f;
                        moving_average_Y = moving_average_Y*0.6f+yhatY*0.4f;
                        //Log.d(TAG,"Queue("+moving_average_X+","+moving_average_Y+")");
                    }

                    Log.d("LATENCY","Gaze Filtered");

                    /**
                     * Plotting Dots
                     * */
                    float inputX = moving_average_X;
                    float inputY = moving_average_Y;
                    DisplayMetrics dm = Fcontext.getResources().getDisplayMetrics();
                    if(calibration_model_exist) {
                        if(calibration_mode_SVR) {
                            //Store Values to run libsvm's SVR
                            float normx = yhatX / (float) dm.widthPixels;
                            float normy = yhatY / (float) dm.heightPixels;
                            int label = 0;
                            appendLog(label + " 1:" + normx + " 2:" + normy, "X");
                            appendLog(label + " 1:" + normx + " 2:" + normy, "Y");
                            svmX.predict(basedir + "X.txt " + basedir + "svmX.model " + basedir + "outX.txt");
                            svmY.predict(basedir + "Y.txt " + basedir + "svmY.model " + basedir + "outY.txt");
                            BufferedReader outputX = new BufferedReader(new FileReader(basedir + "outX.txt"));
                            float outX = Float.parseFloat(outputX.readLine()) * dm.widthPixels;
                            BufferedReader outputY = new BufferedReader(new FileReader(basedir + "outY.txt"));
                            float outY = Float.parseFloat(outputY.readLine()) * dm.heightPixels;
                            Log.d(TAG, "outX: " + outX + " outY: " + outY);
                            calib_X = outX;
                            calib_Y = outY;
                        }
                        else {
                            if(CORNER_CALIBRATION) {
                                // Calcuate Calibration Points
                                float len_X = (float) dm.widthPixels;
                                float len_Y = (float) dm.heightPixels;
                                calib_X = ((inputX - center_offset_X) / (bottom_right_offset_X - top_left_offset_X)) * len_X + center_offset_X;
                                calib_Y = ((inputY - center_offset_Y) / (bottom_right_offset_Y - top_left_offset_Y)) * len_Y + center_offset_Y;
                            }
                            else {
                                //TODO 5- Point Calibration
                                float len_X = (float) dm.widthPixels/2.0f;
                                float len_Y = (float) dm.heightPixels/2.0f;
                                float relx = yhatX-center_mean_X;
                                float rely = yhatY-center_mean_Y;
                                float a = relx;
                                float b = rely;
                                if (relx <=0 && rely <=0 ){
                                    //tlxscale, tlyscale
                                    calib_X = a*tlxscale+len_X;
                                    calib_Y = b*tlyscale+len_Y;
                                }
                                else if (relx>0 && rely<=0){
                                    //trxscale, tryscale
                                    calib_X = a*trxscale+len_X;
                                    calib_Y = b*tryscale+len_Y;
                                }
                                else if (relx<=0 && rely>0){
                                    //blxscale, blyscale
                                    calib_X = a*blxscale+len_X;
                                    calib_Y = b*blyscale+len_Y;
                                }
                                else if (relx>0 && rely>0){
                                    //brxscale, bryscale
                                    calib_X = a*brxscale+len_X;
                                    calib_Y = b*bryscale+len_Y;
                                }
                                if(calib_moving_average){
                                    calib_moving_average_X = calib_X;
                                    calib_moving_average_Y = calib_Y;
                                    calib_moving_average = false;
                                }
                                else {
                                    calib_moving_average_X = calib_moving_average_X*0.6f+calib_X*0.4f;
                                    calib_moving_average_Y = calib_moving_average_Y*0.6f+calib_Y*0.4f;
                                    //Log.d(TAG,"Queue("+moving_average_X+","+moving_average_Y+")");
                                }
                            }
                        }
                        Log.d("MOBED_GazePoint_Calib","x:"+calib_X+" y:"+calib_Y);
                    }
                }
                catch (java.lang.NullPointerException e){
                    Log.e(TAG, "tflite is not working: "+ e);
                    e.printStackTrace();
                }
                catch (Exception e){
                    e.printStackTrace();
                }

                Log.d("MOBED_GazePoint","x : "+String.format("%f", yhatX)+" || y : "+String.format("%f", yhatY));

                /**
                 * MOBED SaveBitmapToFileCache
                 * Made For Debug Purpose you can save bitmap image to /sdcard/CaptureApp directory
                 * Then check how the bitmap data is.
                 * */
//                SharedPreferences sf = LivePreviewActivity.getSf();
//                int count = sf.getInt("count",0);
//                String file0 = "lefteye"+count+".jpg";
//                String file1 = "righteye"+count+".jpg";
//                String file2 = "face"+count+".jpg";
//                SaveBitmapToFileCache(leftBitmap,"/sdcard/CaptureApp/lefteye/",file0);
//                SaveBitmapToFileCache(rightBitmap,"/sdcard/CaptureApp/righteye/",file1);
//                //SaveBitmapToFileCache(faceBitmap,"/sdcard/CaptureApp/face/",file2);
//                Log.d(TAG, "Bitmap saved");
//                LivePreviewActivity.addCount();
            }
            catch (java.lang.IllegalArgumentException e) {
                Log.e(TAG, "java.lang.IllegalArgumentException");
                e.printStackTrace();
            }
            catch (Exception e){
                e.printStackTrace();
            }
            Log.d(TAG, "Bitmap created");

            /**
             * MOBED Calibration Implementation
             * Made For Runtime Calibration
             * 5 - points Calibration (TopLeft, TopRight, BottomLeft, BottomRight, Center)
             * */
            float inputX = moving_average_X;
            float inputY = moving_average_Y;
            if(calibration_flag){
                DisplayMetrics dm = Fcontext.getResources().getDisplayMetrics();
                //for normalization
                float normx = inputX/(float) dm.widthPixels;
                float normy = inputY/(float) dm.heightPixels;
                calibration_button.setVisibility(View.INVISIBLE);
                GraphicOverlay maskOverlay = ((Activity)Fcontext).findViewById(R.id.mask_overlay);
                maskOverlay.setVisibility(View.VISIBLE);
                //if SACCADE this data is not suitable for calibration
                if(isSaccade) continue;
                Button calibration_point = (Button) ((Activity)Fcontext).findViewById(R.id.calibration_point);
                calibration_point.setVisibility(View.VISIBLE);
                RelativeLayout.LayoutParams params = (RelativeLayout.LayoutParams)calibration_point.getLayoutParams();
                TextView calibration_instruction = (TextView) ((Activity)Fcontext).findViewById(R.id.calibration_instruction);
                // Calibration Phase
                if(calibration_phase<FPS*2) {
                    calibration_point.setVisibility(View.INVISIBLE);
                    calibration_instruction.setVisibility(View.VISIBLE);
                    if(CORNER_CALIBRATION) center_mean_X =  center_mean_Y = top_left_mean_X = top_left_mean_Y = bottom_left_mean_X = top_right_mean_Y = 5000;
                    else center_mean_X =  center_mean_Y = top_left_mean_X = top_left_mean_Y = bottom_left_mean_X = top_right_mean_Y = 0;
                    top_right_mean_X = bottom_left_mean_Y = bottom_right_mean_X = bottom_right_mean_Y = 0;
                }
                else if(calibration_phase<FPS*3) {
                    //skip first 10 results (eye movement delay)
                    calibration_instruction.setVisibility(View.INVISIBLE);
                    if (calibration_phase<(FPS*3+SKIP_FRAME)){
                        //calibration on center
                        params.addRule(RelativeLayout.ALIGN_PARENT_TOP, 0);
                        params.addRule(RelativeLayout.ALIGN_PARENT_BOTTOM, 0);
                        params.addRule(RelativeLayout.ALIGN_PARENT_LEFT, 0);
                        params.addRule(RelativeLayout.ALIGN_PARENT_RIGHT, 0);
                        //subject staring at point (dm.heightPixels/2,widthPixels/2) but estimated point is (yhatX,yhatY)
                        appendLog("0.5 1:" + normx + " 2:" + normy, "trainX");
                        appendLog("0.5 1:" + normx + " 2:" + normy, "trainY");
                        center_mean_X+=inputX;
                        center_mean_Y+=inputY;
                    }
                }
                else if(calibration_phase<FPS*4) {
                    if (calibration_phase<(FPS*4+SKIP_FRAME)) {
                        //Top Left
                        params.addRule(RelativeLayout.ALIGN_PARENT_TOP, RelativeLayout.TRUE);
                        params.addRule(RelativeLayout.ALIGN_PARENT_BOTTOM, 0);
                        params.addRule(RelativeLayout.ALIGN_PARENT_LEFT, RelativeLayout.TRUE);
                        params.addRule(RelativeLayout.ALIGN_PARENT_RIGHT, 0);
                        //subject staring at point (0,0) but estimated point is (yhatX,yhatY)
                        appendLog("0 1:"+normx+" 2:"+normy,"trainX");
                        appendLog("0 1:"+normx+" 2:"+normy,"trainY");
                        if(CORNER_CALIBRATION) {
                            if (top_left_mean_X > inputX) top_left_mean_X = inputX;
                            if (top_left_mean_Y > inputY) top_left_mean_Y = inputY;
                        }
                        else {
                            top_left_mean_X+=inputX;
                            top_left_mean_Y+=inputY;
                        }
                    }
                }
                else if(calibration_phase<FPS*5) {
                    if (calibration_phase<(FPS*5+SKIP_FRAME)) {
                        //Top Right
                        params.addRule(RelativeLayout.ALIGN_PARENT_TOP, RelativeLayout.TRUE);
                        params.addRule(RelativeLayout.ALIGN_PARENT_BOTTOM, 0);
                        params.addRule(RelativeLayout.ALIGN_PARENT_LEFT, 0);
                        params.addRule(RelativeLayout.ALIGN_PARENT_RIGHT, RelativeLayout.TRUE);
                        //subject staring at point (dm.widthPixels,0) but estimated point is (yhatX,yhatY)
                        appendLog("1 1:" + normx + " 2:" + normy, "trainX");
                        appendLog("0 1:" + normx + " 2:" + normy, "trainY");
                        if(CORNER_CALIBRATION) {
                            if (top_right_mean_X < inputX) top_right_mean_X = inputX;
                            if (top_right_mean_Y > inputY) top_right_mean_Y = inputY;
                        }
                        else {
                            top_right_mean_X+=inputX;
                            top_right_mean_Y+=inputY;
                        }
                    }
                }
                else if(calibration_phase<FPS*6) {
                    if (calibration_phase<(FPS*6+SKIP_FRAME)) {
                        //Bottom Left
                        params.addRule(RelativeLayout.ALIGN_PARENT_TOP, 0);
                        params.addRule(RelativeLayout.ALIGN_PARENT_BOTTOM, RelativeLayout.TRUE);
                        params.addRule(RelativeLayout.ALIGN_PARENT_LEFT, RelativeLayout.TRUE);
                        params.addRule(RelativeLayout.ALIGN_PARENT_RIGHT, 0);
                        //subject staring at point (0,dm.heightPixels) but estimated point is (yhatX,yhatY)
                        appendLog("0 1:" + normx + " 2:" + normy, "trainX");
                        appendLog("1 1:" + normx + " 2:" + normy, "trainY");
                        if(CORNER_CALIBRATION) {
                            if (bottom_left_mean_X > inputX) bottom_left_mean_X = inputX;
                            if (bottom_left_mean_Y < inputY) bottom_left_mean_Y = inputY;
                        }
                        else {
                            bottom_left_mean_X+=inputX;
                            bottom_left_mean_Y+=inputY;
                        }
                    }
                }
                else if(calibration_phase<FPS*7) {
                    if (calibration_phase<(FPS*7+SKIP_FRAME)) {
                        //Bottom Right
                        params.addRule(RelativeLayout.ALIGN_PARENT_TOP, 0);
                        params.addRule(RelativeLayout.ALIGN_PARENT_BOTTOM, RelativeLayout.TRUE);
                        params.addRule(RelativeLayout.ALIGN_PARENT_LEFT, 0);
                        params.addRule(RelativeLayout.ALIGN_PARENT_RIGHT, RelativeLayout.TRUE);
                        //subject staring at point (dm.heightPixels,widthPixels) but estimated point is (yhatX,yhatY)
                        appendLog("1 1:" + normx + " 2:" + normy, "trainX");
                        appendLog("1 1:" + normx + " 2:" + normy, "trainY");
                        if(CORNER_CALIBRATION) {
                            if (bottom_right_mean_X < inputX) bottom_right_mean_X = inputX;
                            if (bottom_right_mean_Y < inputY) bottom_right_mean_Y = inputY;
                        }
                        else {
                            bottom_right_mean_X+=inputX;
                            bottom_right_mean_Y+=inputY;
                        }
                    }
                }
                else if(calibration_phase<FPS*8) {
                    if(calibration_phase<FPS*8+1) {
                        calibration_flag = false;
                        // TODO Loading GIF and Training SVR and Deploy it
                        // libsvm train option "-s 3 -t 2 -c COST -g GAMMA" will do the magic probably
                        if(calibration_mode_SVR) {
                            // SVR Calibration
                            String svmXdir = basedir + "trainX.txt";
                            String svmYdir = basedir + "trainY.txt";
                            svmX.train("-s 3 -t 2 -c " + COST + " -g " + GAMMA + " " + svmXdir + " " + basedir + "svmX.model");
                            svmY.train("-s 3 -t 2 -c " + COST + " -g " + GAMMA + " " + svmYdir + " " + basedir + "svmY.model");
                        }
                        else {
                            // Calculating
                            center_mean_X = center_mean_X / (float) (FPS - SKIP_FRAME);
                            center_mean_Y = center_mean_Y / (float) (FPS - SKIP_FRAME);
                            if(CORNER_CALIBRATION) {
                                Log.d("MOBED_CalibOffset", "center_mean x:" + center_mean_X + " y:" + center_mean_Y);
                                Log.d("MOBED_CalibOffset", "top_left_mean x:" + top_left_mean_X + " y:" + top_left_mean_Y);
                                Log.d("MOBED_CalibOffset", "top_right_mean x:" + top_right_mean_X + " y:" + top_right_mean_Y);
                                Log.d("MOBED_CalibOffset", "bottom_left_mean x:" + bottom_left_mean_X + " y:" + bottom_left_mean_Y);
                                Log.d("MOBED_CalibOffset", "bottom_right_mean x:" + bottom_right_mean_X + " y:" + bottom_right_mean_Y);
                                // offset values
                                center_offset_X = center_mean_X;
                                center_offset_Y = center_mean_Y;
                                top_left_offset_X = (top_left_mean_X + bottom_left_mean_X) / 2.0f;
                                top_left_offset_Y = (top_left_mean_Y + top_right_mean_Y) / 2.0f;
                                bottom_right_offset_X = (top_right_mean_X + bottom_right_mean_X) / 2.0f;
                                bottom_right_offset_Y = (bottom_left_mean_Y + bottom_right_mean_Y) / 2.0f;
                                center_offset_X = (bottom_right_offset_X + top_left_offset_X) / 2.0f;
                                center_offset_Y = (bottom_right_offset_Y + top_left_offset_Y) / 2.0f;
                                Log.d("MOBED_CalibOffset", "center_offset x:" + center_offset_X + " y:" + center_offset_Y);
                                Log.d("MOBED_CalibOffset", "top_left x:" + top_left_offset_X + " y:" + top_left_offset_Y);
                                Log.d("MOBED_CalibOffset", "bottom_right x:" + bottom_right_offset_X + " y:" + bottom_right_offset_Y);
                            }
                            else {
                                //TODO
                                top_left_mean_X = top_left_mean_X / (float) (FPS - SKIP_FRAME);
                                top_left_mean_Y = top_left_mean_Y / (float) (FPS - SKIP_FRAME);
                                top_right_mean_X = top_right_mean_X / (float) (FPS - SKIP_FRAME);
                                top_right_mean_Y = top_right_mean_Y / (float) (FPS - SKIP_FRAME);
                                bottom_left_mean_X = bottom_left_mean_X / (float) (FPS - SKIP_FRAME);
                                bottom_left_mean_Y = bottom_left_mean_Y / (float) (FPS - SKIP_FRAME);
                                bottom_right_mean_X = bottom_right_mean_X / (float) (FPS - SKIP_FRAME);
                                bottom_right_mean_Y = bottom_right_mean_Y / (float) (FPS - SKIP_FRAME);
                                Log.d("MOBED_CalibOffset", "center_mean x:" + center_mean_X + " y:" + center_mean_Y);
                                Log.d("MOBED_CalibOffset", "top_left_mean x:" + top_left_mean_X + " y:" + top_left_mean_Y);
                                Log.d("MOBED_CalibOffset", "top_right_mean x:" + top_right_mean_X + " y:" + top_right_mean_Y);
                                Log.d("MOBED_CalibOffset", "bottom_left_mean x:" + bottom_left_mean_X + " y:" + bottom_left_mean_Y);
                                Log.d("MOBED_CalibOffset", "bottom_right_mean x:" + bottom_right_mean_X + " y:" + bottom_right_mean_Y);
                                // offset values
                                center_offset_X = center_mean_X;
                                center_offset_Y = center_mean_Y;
                                //translate (cx,cy) to be an origin (0,0)
                                float cx = dm.widthPixels/2.0f;
                                float cy = dm.heightPixels/2.0f;
                                //scaling values
                                tlxscale = Math.abs(cx/(center_offset_X - top_left_mean_X));
                                tlyscale = Math.abs(cy/(center_offset_Y - top_left_mean_Y));
                                trxscale = Math.abs(cx/(top_right_mean_X - center_offset_X));
                                tryscale = Math.abs(cy/(center_offset_Y - top_right_mean_Y));
                                blxscale = Math.abs(cx/(center_offset_X - bottom_left_mean_X));
                                blyscale = Math.abs(cy/(bottom_left_mean_Y - center_offset_Y));
                                brxscale = Math.abs(cx/(bottom_right_mean_X - center_offset_X));
                                bryscale = Math.abs(cy/(bottom_right_mean_Y - center_offset_Y));

                                Log.d("MOBED_CalibOffset", "tl x:" + tlxscale + " y:" + tlyscale);
                                Log.d("MOBED_CalibOffset", "tr x:" + trxscale + " y:" + tryscale);
                                Log.d("MOBED_CalibOffset", "bl x:" + blxscale + " y:" + blyscale);
                                Log.d("MOBED_CalibOffset", "br x:" + brxscale + " y:" + bryscale);
                            }
                        }
                        // Calibration Done
                        calibration_model_exist = true;
                    }
                    else calibration_point.setVisibility(View.INVISIBLE);
                }
                calibration_phase++;
                calibration_point.setLayoutParams(params);
            }
            else {
                calibration_phase=0;
                //Show the Gaze point, Eye Boxes on graphic overlay
                GraphicOverlay maskOverlay = ((Activity)Fcontext).findViewById(R.id.mask_overlay);
                maskOverlay.setVisibility(View.INVISIBLE);
                Button calibration_point = (Button) ((Activity)Fcontext).findViewById(R.id.calibration_point);
                calibration_point.setVisibility(View.INVISIBLE);
                calibration_button.setVisibility(View.VISIBLE);
                calibration_button.setText(R.string.calibration);
                if(!calib_moving_average) {
                    graphicOverlay.add(new FaceGraphic(graphicOverlay, face, yhatX, yhatY, calib_X, calib_Y, calib_moving_average_X, calib_moving_average_Y));
                }
                else {
                    graphicOverlay.add(new FaceGraphic(graphicOverlay, face, yhatX, yhatY, moving_average_X, moving_average_Y, calib_X, calib_Y));
                }

                Log.d("LATENCY","Gaze Painted");
            }
            Log.d("LATENCY","End");
        }
    }

    /**
     * MOBED
     * Made For Debug Purpose
     * */
    public static void SaveBitmapToFileCache(Bitmap bitmap, String strFilePath, String filename) {
        File file = new File(strFilePath);
        if (!file.exists())
            file.mkdirs();
        File fileCacheItem = new File(strFilePath + filename);
        Log.d(TAG, "filename: "+strFilePath + filename);
        FileOutputStream out = null;
        try {
            out = new FileOutputStream(fileCacheItem);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
            out.flush();
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.e(TAG, "Face detection failed " + e);
    }

    /**
     * MOBED appendLog()
     * for libsvm trainset creation
     * */
    public void appendLog(String text, String option) {
        File logFile = new File(basedir+option+".txt");
        if (!logFile.exists()) {
            try {
                if(logFile.createNewFile()){
                    BufferedWriter buf = new BufferedWriter(new FileWriter(logFile, true));
                    buf.close();
                }
            }
            catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        try {
            //BufferedWriter for performance, true to set append to file flag
            BufferedWriter buf = new BufferedWriter(new FileWriter(logFile, true));
            buf.append(text);
            buf.newLine();
            buf.close();
        }
        catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
