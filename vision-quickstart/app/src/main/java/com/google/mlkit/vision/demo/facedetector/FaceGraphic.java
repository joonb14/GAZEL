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

import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.GraphicOverlay.Graphic;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceContour;
import com.google.mlkit.vision.face.FaceLandmark;
import com.google.mlkit.vision.face.FaceLandmark.LandmarkType;

import java.util.Locale;

/**
 * Graphic instance for rendering face position, contour, and landmarks within the associated
 * graphic overlay view.
 */
public class FaceGraphic extends Graphic {
    private static final float EYE_BOX_RATIO = 1.4f;
    private static final float FACE_POSITION_RADIUS = 4.0f;
    private static final float ID_TEXT_SIZE = 30.0f;
    private static final float ID_Y_OFFSET = 40.0f;
    private static final float ID_X_OFFSET = -40.0f;
    private static final float BOX_STROKE_WIDTH = 5.0f;
    private static final int NUM_COLORS = 10;
    private static final int[][] COLORS = new int[][]{
            // {Text color, background color}
            {Color.BLACK, Color.WHITE},
            {Color.WHITE, Color.MAGENTA},
            {Color.BLACK, Color.LTGRAY},
            {Color.WHITE, Color.RED},
            {Color.WHITE, Color.BLUE},
            {Color.WHITE, Color.DKGRAY},
            {Color.BLACK, Color.CYAN},
            {Color.BLACK, Color.YELLOW},
            {Color.WHITE, Color.BLACK},
            {Color.BLACK, Color.GREEN}
    };
    public float leftEyeleft, leftEyetop, leftEyeright, leftEyebottom;
    public float rightEyeleft, rightEyetop, rightEyeright, rightEyebottom;
    public static final int FACE = 1;
    public static final int LEFT_EYEBROW_TOP = 2;
    public static final int LEFT_EYEBROW_BOTTOM = 3;
    public static final int RIGHT_EYEBROW_TOP = 4;
    public static final int RIGHT_EYEBROW_BOTTOM = 5;
    public static final int LEFT_EYE = 6;
    public static final int RIGHT_EYE = 7;
    public static final int UPPER_LIP_TOP = 8;
    public static final int UPPER_LIP_BOTTOM = 9;
    public static final int LOWER_LIP_TOP = 10;
    public static final int LOWER_LIP_BOTTOM = 11;
    public static final int NOSE_BRIDGE = 12;
    public static final int NOSE_BOTTOM = 13;
    public static final int LEFT_CHEEK = 14;
    public static final int RIGHT_CHEEK = 15;

    private final Paint facePositionPaint;
    private final Paint  rightEyePaint,leftEyePaint;
    private volatile Face face;

    FaceGraphic(GraphicOverlay overlay, Face face) {
        super(overlay);

        this.face = face;
        final int selectedColor = Color.WHITE;

        facePositionPaint = new Paint();
        facePositionPaint.setColor(selectedColor);

        leftEyePaint = new Paint();
        leftEyePaint.setColor(Color.RED);
        leftEyePaint.setStyle(Paint.Style.STROKE);
        leftEyePaint.setStrokeWidth(BOX_STROKE_WIDTH);

        rightEyePaint = new Paint();
        rightEyePaint.setColor(Color.BLUE);
        rightEyePaint.setStyle(Paint.Style.STROKE);
        rightEyePaint.setStrokeWidth(BOX_STROKE_WIDTH);
    }

    /**
     * Draws the face annotations for position on the supplied canvas.
     */
    @Override
    public void draw(Canvas canvas) {
        Face face = this.face;
        if (face == null) {
            return;
        }

        // Draws a circle at the position of the detected face, with the face's track id below.
        float x = translateX(face.getBoundingBox().centerX());
        float y = translateY(face.getBoundingBox().centerY());
        canvas.drawCircle(x, y, FACE_POSITION_RADIUS, facePositionPaint);

        // Calculate positions.
        float left = x - scale(face.getBoundingBox().width() / 2.0f);
        float top = y - scale(face.getBoundingBox().height() / 2.0f);
        float right = x + scale(face.getBoundingBox().width() / 2.0f);
        float bottom = y + scale(face.getBoundingBox().height() / 2.0f);
        float lineHeight = ID_TEXT_SIZE + BOX_STROKE_WIDTH;
        float yLabelOffset = -lineHeight;

        // Decide color based on face ID
        int colorID = (face.getTrackingId() == null)
                ? 0 : Math.abs(face.getTrackingId() % NUM_COLORS);


        FaceContour leftEyeContour = face.getContour(LEFT_EYE);
        FaceContour rightEyeContour = face.getContour(RIGHT_EYE);
        float righteye_leftx = translateX(rightEyeContour.getPoints().get(0).x);
        float righteye_lefty = translateY(rightEyeContour.getPoints().get(0).y);
        float righteye_rightx = translateX(rightEyeContour.getPoints().get(7).x);
        float righteye_righty = translateY(rightEyeContour.getPoints().get(7).y);
        float lefteye_leftx = translateX(leftEyeContour.getPoints().get(0).x);
        float lefteye_lefty = translateY(leftEyeContour.getPoints().get(0).y);
        float lefteye_rightx = translateX(leftEyeContour.getPoints().get(7).x);
        float lefteye_righty = translateY(leftEyeContour.getPoints().get(7).y);
        float righteye_centerx = (righteye_leftx + righteye_rightx)/2.0f;
        float righteye_centery = (righteye_lefty + righteye_righty)/2.0f;
        float lefteye_centerx = (lefteye_leftx + lefteye_rightx)/2.0f;
        float lefteye_centery = (lefteye_lefty + lefteye_righty)/2.0f;
        float lefteyeboxsize = (lefteye_centerx-lefteye_leftx)*EYE_BOX_RATIO;
        float righteyeboxsize = (righteye_centerx-righteye_leftx)*EYE_BOX_RATIO;
//        canvas.drawCircle(righteye_centerx,righteye_centery,FACE_POSITION_RADIUS, rightEyePaint);
//        canvas.drawCircle(lefteye_centerx,lefteye_centery,FACE_POSITION_RADIUS, leftEyePaint);
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

        // Draw facial landmarks
        drawFaceLandmark(canvas, FaceLandmark.LEFT_EYE);
        drawFaceLandmark(canvas, FaceLandmark.RIGHT_EYE);
        drawFaceLandmark(canvas, FaceLandmark.LEFT_CHEEK);
        drawFaceLandmark(canvas, FaceLandmark.RIGHT_CHEEK);
    }

    private void drawFaceLandmark(Canvas canvas, @LandmarkType int landmarkType) {
        FaceLandmark faceLandmark = face.getLandmark(landmarkType);
        if (faceLandmark != null) {
            canvas.drawCircle(
                    translateX(faceLandmark.getPosition().x),
                    translateY(faceLandmark.getPosition().y),
                    FACE_POSITION_RADIUS,
                    facePositionPaint);
        }
    }
}
