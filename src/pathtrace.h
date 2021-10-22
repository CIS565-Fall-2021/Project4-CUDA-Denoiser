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
	glm::ivec2 offset[25];
	int stepWidth;

	//float sigma2RT, sigma2N, sigma2X;
};

enum GBufferType
{
	COLOR = 0,
	POSITION,
	NORMAL
};

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration, int max);
void showGBuffer(uchar4 *pbo, GBufferType type);
void showImage(uchar4 *pbo, int iter);
void showDenoisedImage(uchar4* pbo, int iter, Denoise denoise);

void cudaStartTime();
void cudaEndTime();
