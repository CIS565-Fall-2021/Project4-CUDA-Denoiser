#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration, bool sortByMaterial, bool cachFirstBounce, bool stochasticAA, bool depthOfField, 
					bool boundingVolumeCulling);
performanceAnalysis::PerformanceTimer& timer();
void showGBuffer(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter);
void denoisePathTrace(uchar4* pbo, int iter, int filter_size, float c_sigma, float n_sigma, float p_sigma);
