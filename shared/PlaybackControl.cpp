/*
 *  PlaybackControl.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/9/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "PlaybackControl.h"

PlaybackControl::PlaybackControl() {}

void PlaybackControl::setCurrentFrame(int x)
{
	m_currentFrame = x;
}

void PlaybackControl::setFrameRange(int mn, int mx)
{
	m_rangeMin = mn;
	m_rangeMax = mx;
	m_currentFrame = mn;
}

int PlaybackControl::currentFrame() const
{
	return m_currentFrame;
}

int PlaybackControl::rangeMin() const
{
	return m_rangeMin;
}

int PlaybackControl::rangeMax() const
{
	return m_rangeMax;
}

void PlaybackControl::disableControl() {}
void PlaybackControl::enableControl() {}