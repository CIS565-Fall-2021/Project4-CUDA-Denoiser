#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene, float a, float b, float c, float* gausKernel, float filterSize);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter);
void showDenoise(uchar4 *pbo, int iter);

