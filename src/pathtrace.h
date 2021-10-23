#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene, float* gausKernel);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter);
void showDenoise(uchar4 *pbo, int iter);

bool DenoiseImage(int resolutionX, int resolutionY, int iteration, int filterSize, float colWeight, float norWeight, float posWeight);