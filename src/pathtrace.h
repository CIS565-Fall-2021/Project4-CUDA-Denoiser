#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration, int viewChoice);
void showGBuffer(uchar4* pbo, int viewChoice);
void showImage(uchar4* pbo, int iter);
void showImageDenoised(uchar4* pbo, int iter, float phi_c, float phi_n, float phi_p, int filterSize);
