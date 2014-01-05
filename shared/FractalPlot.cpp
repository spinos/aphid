/*
 *  FractalPlot.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/5/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "FractalPlot.h"

FractalPlot::FractalPlot() {}
FractalPlot::~FractalPlot() {}

float FractalPlot::getNoise(float u, unsigned frequency, float lod, unsigned seed) const
{
	int head = seed & 26431;
	const int level = lod * 5;
	const float lft = lod * 5 - level;
	float length = frequency;
	float r = getValue(u, head, length);
	head += length;
	float scale = .5f;
	length *= 2;
	for(int i = 0; i <= level; i++) {
		if(lft > 0.f && i == level) scale *= lft;
		r += getValue(u, head, length) * scale;
		head += length;
		length *= 2;
		scale *= 0.5f;
	}
	return r;
}

float FractalPlot::getValue(float u, unsigned head, unsigned length) const
{
	const float coord = u * length;
	const int i = coord;
	const float portion = coord - i;
	return sample(head + i) * (1.f - portion) + sample(head + i + 1) * portion;
}