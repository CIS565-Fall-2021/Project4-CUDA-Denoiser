#pragma once

#include <vector>
#include "scene.h"
// #include "main.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
// void pathtrace(uchar4 *pbo, int frame, int iteration);
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo, float ui_normalWeight, float ui_positionWeight, float ui_colorWeight);
void showImage(uchar4 *pbo, int iter);
void showDenoised(uchar4 *pbo, int iter);
void denoise(int ui_filterSize, float ui_normalWeight, float ui_positionWeight, float ui_colorWeight);
void denoiseInit(Scene *scene);
void denoiseFree();
void setBufferToDenoised();
