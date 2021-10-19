#pragma once

#include <vector>
#include "scene.h"

// Different display mode for the Gbuffer
#define GBUFFER_TIME 0
#define GBUFFER_NORMAL 1
#define GBUFFER_POSITION 2

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration, bool denoise, int filterSize, float colorWeight, float normalWeight, float positionWeight);
//void denoise(int filterSize, float colorWeight, float normalWeight, float positionWeight);
void showGBuffer(uchar4 *pbo, int mode);
void showImage(uchar4 *pbo, int iter);
