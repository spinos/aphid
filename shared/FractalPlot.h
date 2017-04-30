/*
 *  FractalPlot.h
 *  mallard
 *
 *  Created by jian zhang on 1/5/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <NoisePlot.h>
class FractalPlot : public NoisePlot {
public:
	FractalPlot();
	virtual ~FractalPlot();
	float getNoise(float u, unsigned frequency, float lod, unsigned seed) const;
protected:

private:
	float getValue(float u, unsigned head, unsigned length) const;
};