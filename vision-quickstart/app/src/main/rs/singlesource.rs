// File: singlesource.rs
// From Android RenderScript Example
// https://developer.android.com/guide/topics/renderscript/compute
// If you want Histogram Equalization look at: https://medium.com/@qhutch/android-simple-and-fast-image-processing-with-renderscript-2fa8316273e1

#pragma version(1)
#pragma rs java_package_name(com.google.mlkit.vision.demo.facedetector)

static const float3 weight = {0.299f, 0.587f, 0.114f};

uchar4 RS_KERNEL greyscale(uchar4 in) {
  const float4 inF = rsUnpackColor8888(in);
  const float3 outF = dot(inF.rgb, weight);
  return rsPackColorTo8888(outF);
}

void process(rs_allocation inputImage, rs_allocation outputImage) {
   rsForEach(greyscale, inputImage, outputImage);
 }