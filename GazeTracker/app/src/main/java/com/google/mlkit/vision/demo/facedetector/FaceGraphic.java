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

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.util.DisplayMetrics;
import android.util.Log;

import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.GraphicOverlay.Graphic;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceContour;
import com.google.mlkit.vision.face.FaceLandmark;
import com.google.mlkit.vision.face.FaceLandmark.LandmarkType;

import java.util.List;
import java.util.Locale;

/**
 * Graphic instance for rendering eye position and Gaze point
 * graphic overlay view.
 */
public class FaceGraphic extends Graphic {
    private String TAG = "MOBED_FaceGraphic";
    private boolean showEyes = true;
    private static final float EYE_BOX_RATIO = 1.4f;
    private static final float FACE_POSITION_RADIUS = 30.0f;
    private static final float BOX_STROKE_WIDTH = 5.0f;
    public float leftEyeleft, leftEyetop, leftEyeright, leftEyebottom;
    public float rightEyeleft, rightEyetop, rightEyeright, rightEyebottom;

    private final Paint  rightEyePaint,leftEyePaint, gazePointPaint, movingAveragePaint, calibPaint;
    private volatile Face face;
    private float yhatx;
    private float yhaty;
    private float x, y, moving_average_X, moving_average_Y,calib_X,calib_Y;

    FaceGraphic(GraphicOverlay overlay, Face face, float yhatx, float yhaty, float moving_average_X, float moving_average_Y, float calib_X, float calib_Y) {
        super(overlay);

        this.face = face;
        this.yhatx = yhatx;
        this.yhaty = yhaty;
        this.moving_average_X = moving_average_X;
        this.moving_average_Y = moving_average_Y;
        this.calib_X = calib_X;
        this.calib_Y = calib_Y;
        final int selectedColor = Color.RED;

        gazePointPaint = new Paint();
        gazePointPaint.setColor(selectedColor);

        movingAveragePaint = new Paint();
        movingAveragePaint.setColor(Color.BLUE);

        calibPaint = new Paint();
        calibPaint.setColor(Color.GREEN);

        leftEyePaint = new Paint();
        leftEyePaint.setColor(Color.WHITE);
        leftEyePaint.setStyle(Paint.Style.STROKE);
        leftEyePaint.setStrokeWidth(BOX_STROKE_WIDTH);

        rightEyePaint = new Paint();
        rightEyePaint.setColor(Color.BLUE);
        rightEyePaint.setStyle(Paint.Style.STROKE);
        rightEyePaint.setStrokeWidth(BOX_STROKE_WIDTH);
    }

    /**
     * Draws the eye positions and gaze point.
     */
    @Override
    public void draw(Canvas canvas) {
        Face face = this.face;
        if (face == null) {
            return;
        }
        Log.d(TAG, "Canvas Width: "+canvas.getWidth()+" Height: "+ canvas.getHeight());

        // Draws a circle at the position of the estimated gaze point
        //DisplayMetrics dm = getApplicationContext().getResources().getDisplayMetrics();
        //Log.d(TAG, "Display Metric w/h: " + width+"/"+height);
        //
        // TODO filter and smooth out values so that gaze point movement looks like smooth pursuit
        //
        int width = canvas.getWidth();
        int height = canvas.getHeight();
        if (yhatx>=width) x = width;
        else if(yhatx<0) x =0;
        else x = yhatx;
        if (yhaty>=height) y = height;
        else if(yhaty<0) y =0;
        else y = yhaty;
        canvas.drawCircle(x, y, FACE_POSITION_RADIUS, gazePointPaint);
        if (moving_average_X>=width) x = width;
        else if(moving_average_X<0) x =0;
        else x = moving_average_X;
        if (moving_average_Y>=height) y = height;
        else if(moving_average_Y<0) y =0;
        else y = moving_average_Y;
        canvas.drawCircle(x, y, FACE_POSITION_RADIUS, movingAveragePaint);
        if (calib_X>=width) x = width;
        else if(calib_X<0) x =0;
        else x = calib_X;
        if (calib_Y>=height) y = height;
        else if(calib_Y<0) y =0;
        else y = calib_Y;
        canvas.drawCircle(x, y, FACE_POSITION_RADIUS, calibPaint);

        if(showEyes) {
            try {
                List<PointF> leftEyeContour = face.getContour(FaceContour.LEFT_EYE).getPoints();
                List<PointF> rightEyeContour = face.getContour(FaceContour.RIGHT_EYE).getPoints();
                float righteye_leftx = translateX(rightEyeContour.get(0).x);
                float righteye_lefty = translateY(rightEyeContour.get(0).y);
                float righteye_rightx = translateX(rightEyeContour.get(8).x);
                float righteye_righty = translateY(rightEyeContour.get(8).y);
                float lefteye_leftx = translateX(leftEyeContour.get(0).x);
                float lefteye_lefty = translateY(leftEyeContour.get(0).y);
                float lefteye_rightx = translateX(leftEyeContour.get(8).x);
                float lefteye_righty = translateY(leftEyeContour.get(8).y);

                float righteye_centerx = (righteye_leftx + righteye_rightx) / 2.0f;
                float righteye_centery = (righteye_lefty + righteye_righty) / 2.0f;
                float lefteye_centerx = (lefteye_leftx + lefteye_rightx) / 2.0f;
                float lefteye_centery = (lefteye_lefty + lefteye_righty) / 2.0f;
                float lefteyeboxsize = (lefteye_centerx - lefteye_leftx) * EYE_BOX_RATIO;
                float righteyeboxsize = (righteye_centerx - righteye_leftx) * EYE_BOX_RATIO;
                leftEyeleft = lefteye_centerx - lefteyeboxsize;
                leftEyetop = lefteye_centery + lefteyeboxsize;
                leftEyeright = lefteye_centerx + lefteyeboxsize;
                leftEyebottom = lefteye_centery - lefteyeboxsize;
                rightEyeleft = righteye_centerx - righteyeboxsize;
                rightEyetop = righteye_centery + righteyeboxsize;
                rightEyeright = righteye_centerx + righteyeboxsize;
                rightEyebottom = righteye_centery - righteyeboxsize;
                canvas.drawRect(rightEyeleft, rightEyetop, rightEyeright, rightEyebottom, rightEyePaint);
                canvas.drawRect(leftEyeleft, leftEyetop, leftEyeright, leftEyebottom, leftEyePaint);
            }
            catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
