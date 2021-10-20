#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration, int viewChoice);
void showGBuffer(uchar4* pbo, int viewChoice);
void showImage(uchar4* pbo, int iter);
