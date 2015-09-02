/*
 *  AFrameRange.cpp
 *  
 *
 *  Created by jian zhang on 7/2/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "AFrameRange.h"

int AFrameRange::FirstFrame = 0;
int AFrameRange::LastFrame = 0;
int AFrameRange::SamplesPerFrame = 1;
float AFrameRange::FramesPerSecond = 25.f;

AFrameRange::AFrameRange() {}
AFrameRange::~AFrameRange() {}
void AFrameRange::reset()
{
	FirstFrame = 0;
	LastFrame = 0;
	SamplesPerFrame = 1;
}

bool AFrameRange::isValid() const
{ return (LastFrame > FirstFrame); }

int AFrameRange::numFramesInRange() const
{ return LastFrame - FirstFrame + 1; }

