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
protected:
	float sample(unsigned idx) const;
	unsigned numSamples() const;	
private:
	float * m_samps;
	unsigned m_numSamps;
};