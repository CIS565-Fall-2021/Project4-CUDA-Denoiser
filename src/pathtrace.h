#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
int  pathtrace(int frame, int iteration, int lastIter);
void showGBuffer(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter, bool ui_denoise);

void denoise(const int filterSize, const float cPhi, const float pPhi, const float nPhi);

