#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void run_denoiser(float c_phi, float n_phi, float p_phi, int filter_size);
void showGBuffer(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter);
void show_denoised_image(uchar4 *pbo, int iter);
