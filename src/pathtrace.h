#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration, bool denoise);
void showGBuffer(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter, bool denoise);
void denoiseImage(float filterSize, float colorW, float norW, float posW);
