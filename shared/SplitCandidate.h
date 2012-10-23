/*
 *  SplitCandidate.h
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

class SplitCandidate {
public:
	SplitCandidate();
	
	void setPos(float val);
	void setAxis(int val);
	
	float getPos() const;
	int getAxis() const;
	void verbose() const;
	
	static int Dimension;
	float m_pos;
	int m_axis;
};