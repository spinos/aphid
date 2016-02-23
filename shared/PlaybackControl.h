/*
 *  PlaybackControl.h
 *  mallard
 *
 *  Created by jian zhang on 10/9/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseState.h>

namespace aphid {

class PlaybackControl : public BaseState {
public:
	PlaybackControl();
	
	int rangeMin() const;
	int rangeMax() const;
	int rangeLength() const;
	int currentFrame() const;
	virtual int playbackMin() const;
	virtual int playbackMax() const;
	void setCurrentFrame(int x);
	
	virtual void setFrameRange(int mn, int mx);
	
	virtual void enable();
	virtual void disable();

private:
	int m_rangeMin, m_rangeMax, m_currentFrame;
};

}