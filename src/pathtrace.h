#pragma once

#include <vector>
#include "scene.h"

struct Denoise
{
	int kernelSize;
	float positionWeight;
	float colorWeight;
	float normalWeight;
	float kernel[25];
	glm::vec2 offset[25];
	int stepWidth;
};

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter);
void showDenoisedImage(uchar4* pbo, int iter, Denoise denoise);
