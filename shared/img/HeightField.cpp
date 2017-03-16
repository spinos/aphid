/*
 *  HeightField.cpp
 *  
 *
 *  Created by jian zhang on 3/15/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "HeightField.h"

namespace aphid {

namespace img {

HeightField::Profile HeightField::GlobalHeightFieldProfile;

HeightField::HeightField()
{}

HeightField::HeightField(const SignalTyp & inputSignal) :
GaussianPyramid<float>(inputSignal)
{
	for(int i=0;i<numLevels();++i) {
		computeDerivative(m_derivative[i], 0, i);
	}
}

HeightField::~HeightField()
{}

float HeightField::sampleHeight(BoxSampleProfile<float> * prof) const
{ 
	float val = sample(prof);
	GlobalHeightFieldProfile.valueToHeight(val);
	return val;
}

const HeightField::SignalTyp & HeightField::levelDerivative(int level) const
{ return m_derivative[level]; }

const BoundingBox & HeightField::calculateBBox()
{
	float ymin = 1e8f;
	float ymax = -1e8f;
	getMinMax(ymin, ymax, 0);
	GlobalHeightFieldProfile.valueToHeight(ymin);
	GlobalHeightFieldProfile.valueToHeight(ymax);
	float w = inputSignalNumCols();
	float h = inputSignalNumRows();
	m_bbox.setMin(0.f, ymin, 0.f);
	m_bbox.setMax(w, ymax, h);
	return m_bbox;
}

const BoundingBox & HeightField::getBBox() const
{ return m_bbox; }

}

}
