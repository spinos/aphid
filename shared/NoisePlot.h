/*
 *  NoisePlot.h
 *  aphid
 *
 *  Created by jian zhang on 1/5/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
class NoisePlot {
public:
	NoisePlot();
	virtual ~NoisePlot();
	void clear();
	void createPlot(unsigned n);
	void computePlot(unsigned seed);
	float getNoise(float u, float lod) const;
private:
	unsigned getGridSize(float lod) const;
	float getLevel(float u, unsigned ng) const;
	float getFull(float u) const;
private:
	float * m_samps;
	unsigned m_numSamps;
};