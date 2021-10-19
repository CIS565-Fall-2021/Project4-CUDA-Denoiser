#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);

void denoiseInit(Scene *scene);
void denoiseFree();
void denoise(int filterSize, float c_phi, float n_phi, float p_phi);

void showGBuffer(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter);
void showDenoise(uchar4* pbo, int iter);
