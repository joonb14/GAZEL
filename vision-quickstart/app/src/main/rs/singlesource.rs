// File: singlesource.rs

#pragma version(1)
#pragma rs java_package_name(com.google.mlkit.vision.demo.facedetector)

static const float3 weight = {0.299f, 0.587f, 0.114f};

float RS_KERNEL greyscale(uchar4 in) {
    const float4 inF = rsUnpackColor8888(in);
    float Y = inF.r*0.299f + inF.g*0.587f+ inF.b*0.114f;
    return Y;
}
