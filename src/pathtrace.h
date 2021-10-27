#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration, bool denoise);
void showGBuffer(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter);
void denoise(int filter_size, float color_weight, float position_weight, float normal_weight);
