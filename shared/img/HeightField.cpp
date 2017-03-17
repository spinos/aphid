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

Array3<float> HeightField::InitialValues[6];

HeightField::HeightField()
{}

HeightField::~HeightField()
{}

void HeightField::create(const SignalTyp & inputSignal)
{
	GaussianPyramid<float>::create(inputSignal);

	for(int i=0;i<numLevels();++i) {
		computeDerivative(m_derivative[i], 0, i);
	}
}

float HeightField::sampleHeight(BoxSampleProfile<float> * prof) const
{ 
	float val = sample(prof);
	GlobalHeightFieldProfile.valueToHeight(val);
	return val;
}

const HeightField::SignalTyp & HeightField::levelDerivative(int level) const
{ return m_derivative[level]; }

void HeightField::SetGlobalProfile(float zeroValue, float minHeight, float maxHeight)
{
	GlobalHeightFieldProfile._zeroHeightValue = zeroValue;
	GlobalHeightFieldProfile._minHeight2 = minHeight * 2.f;
	GlobalHeightFieldProfile._maxHeight2 = maxHeight * 2.f;
	GlobalHeightFieldProfile._heightRange = .29f * (maxHeight - minHeight);
	int w = 32;
	for(int i=0;i<6;++i) {
		InitialValues[i].create(w, w, 1);
		InitialValues[i].rank(0)->set(zeroValue);
		w <<= 1;
	}
}

const Array3<float> & HeightField::InitialValueAtLevel(int level)
{ return InitialValues[level - 5]; }

Float2 HeightField::sampleHeightDerivative(BoxSampleProfile<float> * prof) const
{
	Float2 r(0.f, 0.f);
	if(prof->isTexcoordOutofRange() ) {
		return r;
	}
	
	const SignalSliceTyp * sliceU = levelDerivative(prof->_loLevel).rank(0);
	
	const int & m = sliceU->numRows();
	const int & n = sliceU->numCols();
	
	int u = prof->_uCoord * n;
	int v = prof->_vCoord * m;

	float du = sliceU->column(u)[v];
	if(du < 0.f) {
		r.x = du;
		r.y = -du;
	} else {
		r.x = -du;
		r.y = du;
	}
	const SignalSliceTyp * sliceV = levelDerivative(prof->_loLevel).rank(1);
	float dv = sliceV->column(u)[v];
	if(dv < r.x) {
		r.x = dv;
	}
	if(dv > r.y) {
		r.y = dv;
	}
	r.x *= GlobalHeightFieldProfile._heightRange;
	r.y *= GlobalHeightFieldProfile._heightRange;
	return r;
}

void HeightField::setRange(const float & w)
{
	m_range.x = w;
	m_range.y = w * aspectRatio();
	m_sampleSize = m_range.x / levelSignal(0).numCols();
}

const float & HeightField::sampleSize() const
{ return m_sampleSize; }

void HeightField::verbose() const
{
	GaussianPyramid<float>::verbose();
	std::cout<<"\n HeightField "<<m_range.x<<" x "<<m_range.y
		<<"\n sample size "<<m_sampleSize;
}

}

}
