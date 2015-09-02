#ifndef AFRAMERANGE_H
#define AFRAMERANGE_H

/*
 *  AFrameRange.h
 *  
 *
 *  Created by jian zhang on 7/2/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

class AFrameRange {
public:
	AFrameRange();
	virtual ~AFrameRange();
	void reset();
	bool isValid() const;
	int numFramesInRange() const;
    static float FramesPerSecond;
	static int FirstFrame;
	static int LastFrame;
	static int SamplesPerFrame;
};
#endif        //  #ifndef AFRAMERANGE_H
