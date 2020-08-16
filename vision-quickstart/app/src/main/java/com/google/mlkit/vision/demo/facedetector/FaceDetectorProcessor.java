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

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.PointF;
import androidx.renderscript.Allocation;
import androidx.renderscript.RenderScript;
import android.util.Log;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.Task;
import com.google.firebase.FirebaseApp;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.LivePreviewActivity;
import com.google.mlkit.vision.demo.VisionProcessorBase;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceContour;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceLandmark;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Face Detector Demo.
 */
public class FaceDetectorProcessor extends VisionProcessorBase<List<Face>> {
    private static final String TAG = "MOBED_FaceDetector";
    public static Interpreter tflite;
    private int resolution = 64;
    private int grid_size = 50;
    public static Bitmap image;
    private final FaceDetector detector;
    public float leftEyeleft, leftEyetop, leftEyeright, leftEyebottom;
    public float rightEyeleft, rightEyetop, rightEyeright, rightEyebottom;
    private RenderScript RS;
    private ScriptC_singlesource script;
    private static final float EYE_BOX_RATIO = 1.4f;

    private float[][][][] left_4d, right_4d, lefteye_grid, righteye_grid;

    private float yhatX =0, yhatY=0;

    public FaceDetectorProcessor(Context context, FaceDetectorOptions options ) {
        super(context);
        Log.v(MANUAL_TESTING_LOG, "Face detector options: " + options);
        detector = FaceDetection.getClient(options);
        RS = RenderScript.create(context);
        script = new ScriptC_singlesource(RS);
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
        for (Face face : faces) {
            //MOBED
            //This is how you get coordinates, and crop left and right eye
            //Look at https://firebase.google.com/docs/ml-kit/detect-faces#example_2_face_contour_detection for details.
            //We specifically used Eye Contour's point 0 and 8.
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
            try {
                Bitmap leftBitmap=Bitmap.createBitmap(image, (int)leftEyeleft,(int)leftEyebottom,(int)(lefteyeboxsize*2), (int)(lefteyeboxsize*2));
                Bitmap rightBitmap=Bitmap.createBitmap(image, (int)rightEyeleft,(int)rightEyebottom,(int)(righteyeboxsize*2), (int)(righteyeboxsize*2));
                if (leftBitmap.getHeight() > resolution){
                    leftBitmap = Bitmap.createScaledBitmap(leftBitmap, resolution,resolution,false);
                }
                if (rightBitmap.getHeight() > resolution){
                    rightBitmap = Bitmap.createScaledBitmap(rightBitmap, resolution,resolution,false);
                }
                //Renderscript converts RGBA value to YUV's Y value.
                //After RenderScript, Y value will be stored in Red pixel value
                Allocation inputAllocation = Allocation.createFromBitmap( RS, leftBitmap);
                Allocation outputAllocation = Allocation.createTyped( RS, inputAllocation.getType());
                script.invoke_process(inputAllocation, outputAllocation);
                outputAllocation.copyTo(leftBitmap);
                inputAllocation = Allocation.createFromBitmap( RS, rightBitmap);
                outputAllocation = Allocation.createTyped( RS, inputAllocation.getType());
                script.invoke_process(inputAllocation, outputAllocation);
                outputAllocation.copyTo(rightBitmap);
                if (leftBitmap.getHeight() < resolution){
                    leftBitmap = Bitmap.createScaledBitmap(leftBitmap, resolution,resolution,false);
                }
                if (rightBitmap.getHeight() < resolution){
                    rightBitmap = Bitmap.createScaledBitmap(rightBitmap, resolution,resolution,false);
                }
                int[] eye = new int[resolution*resolution];

                /**
                 * Left Eye
                 * */
                leftBitmap.getPixels(eye,0,resolution,0,0,resolution,resolution);
                left_4d = new float[1][resolution][resolution][1];
                for(int y = 0; y < resolution; y++) {
                    for (int x = 0; x < resolution; x++) {
                        int index = y * resolution + x;
                        left_4d[0][y][x][0] = ((eye[index] & 0x00FF0000) >> 16)/(float)255;
                    }
                }
                /**
                 * Right Eye
                 * */
                rightBitmap.getPixels(eye,0,resolution,0,0,resolution,resolution);
                right_4d = new float[1][resolution][resolution][1];
                for(int y = 0; y < resolution; y++) {
                    for (int x = 0; x < resolution; x++) {
                        int index = y * resolution + x;
                        right_4d[0][y][x][0] = ((eye[index] & 0x00FF0000) >> 16)/(float)255;
                    }
                }

                /**
                 * Eye Grids
                 * */
                int image_width = image.getWidth();
                int image_height = image.getHeight();
                //left, bottom, width, height
                float w_start = Math.round(grid_size*(leftEyeleft/(float)image_width));
                float h_start = Math.round(grid_size*(leftEyebottom/(float)image_height));
                float w_num = Math.round(grid_size*((2*lefteyeboxsize)/(float)image_width));
                float h_num = Math.round(grid_size*((2*lefteyeboxsize)/(float)image_height));

                lefteye_grid = new float[1][grid_size][grid_size][1];
                for(int h=0; h<grid_size; h++){
                    for(int w=0; w<grid_size; w++) {
                        if (w>=w_start && w<=w_start+w_num && h>=h_start && h<=h_start+h_num){
                            lefteye_grid[0][h][w][0] = 1;
                        }
                        else lefteye_grid[0][h][w][0] = 0;
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
                            righteye_grid[0][h][w][0] = 1;
                        }
                        else righteye_grid[0][h][w][0] = 0;
                    }
                }

                /**
                 * Wrap them up to use them as input for TensorFlow Lite model
                 * */
                //For jw_model.tflite
                //float[][][][][] inputs = new float[][][][][]{left_4d, righteye_grid, right_4d, lefteye_grid};
                //For ykmodel.tflite
                float[][][][][] inputs = new float[][][][][]{righteye_grid, left_4d, right_4d, lefteye_grid};

                // To use multiple input and multiple output you must use the Interpreter.runForMultipleInputsOutputs()
                float[][] output = new float[1][2];
                Map<Integer, Object> outputs = new HashMap<>();
                outputs.put(0, output);
                try {
                    tflite.runForMultipleInputsOutputs(inputs, outputs);
                    //The output x,y will be stored to below variables
                    yhatX = output[0][0];
                    yhatY = output[0][1];
                    Log.d(TAG, "tflite is working!");
                }
                catch (java.lang.NullPointerException e){
                    Log.e(TAG, "tflite is not working: "+ e);
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
//                SaveBitmapToFileCache(leftBitmap,"/sdcard/CaptureApp/lefteye/",file0);
//                SaveBitmapToFileCache(rightBitmap,"/sdcard/CaptureApp/righteye/",file1);
//                Log.d(TAG, "Bitmap saved");
//                LivePreviewActivity.addCount();
            }
            catch (java.lang.IllegalArgumentException e) {
                Log.e(TAG, "java.lang.IllegalArgumentException");
                e.printStackTrace();
            }
            Log.d(TAG, "Bitmap created");

            //Show the Gaze point, Eye Boxes on graphic overlay
            graphicOverlay.add(new FaceGraphic(graphicOverlay, face, yhatX, yhatY));
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
}
