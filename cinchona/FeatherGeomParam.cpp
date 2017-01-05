/*
 *  FeatherGeomParam.cpp
 *  cinchona
 *
 *  Created by jian zhang on 1/3/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "FeatherGeomParam.h"
#include <math/Vector3F.h>
#include <math/linspace.h>
#include <gpr/GPInterpolate.h>

using namespace aphid;

FeatherGeomParam::FeatherGeomParam() 
{
	_numTotalF = 0;
	for(int i=0;i<3;++i) {
		_numFPerSeg[i] = 0;
		_xsPerSeg[i] = NULL;
		_psPerSeg[i] = NULL;
	}
	for(int i=0;i<4;++i) {
		_chordLength[i] = -1.f;
		_thickness[i] = -1.f;
	}
	m_chordInterp = new gpr::GPInterpolate<float>();
	m_chordInterp->create(4,1,1);
	m_chordInterp->setFilterLength(.33f);
	m_thicknessInterp = new gpr::GPInterpolate<float>();
	m_thicknessInterp->create(4,1,1);
	m_thicknessInterp->setFilterLength(.33f);
}

FeatherGeomParam::~FeatherGeomParam()
{
	for(int i=0;i<3;++i) {
		if(_xsPerSeg[i]) delete[] _xsPerSeg[i];
		if(_psPerSeg[i]) delete[] _psPerSeg[i];
	}
	delete m_chordInterp;
	delete m_thicknessInterp;
	
}

void FeatherGeomParam::set(const int * nps,
						const float * chords,
						const float * ts)
{
	_changed = false;
	_numTotalF = 0;
	for(int i=0;i<3;++i) {
		const int nf = nps[i];
		if(nf != _numFPerSeg[i]) {
			_changed = true;
			_numFPerSeg[i] = nf;
			 
			if(_xsPerSeg[i]) {
				delete[] _xsPerSeg[i];
			}
			_xsPerSeg[i] = new float[nf];
			
/// at center of each segment
			linspace_center<float>(_xsPerSeg[i], 0.f, 1.f, nf);
			
			if(_psPerSeg[i]) {
				delete[] _psPerSeg[i];
			}
			_psPerSeg[i] = new Vector3F[nf];
		}
		_numTotalF += nf;
		
	}
	
	float longestC = 0.f;
	bool chordChanged = false;
	for(int i=0;i<4;++i) {
		if(longestC < chords[i]) {
			longestC = chords[i];
		}
		if(chords[i] != _chordLength[i]) {
			_changed = true;
			chordChanged = true;
			_chordLength[i] = chords[i];
		}
	}

	if(chordChanged) {
		_longestChord = longestC;
		learnChord();
	}
	
	bool thicknessChanged = false;
	for(int i=0;i<4;++i) {
		if(ts[i] != _thickness[i]) {
			_changed = true;
			thicknessChanged = true;
			_thickness[i] = ts[i];
		}
	}

	if(thicknessChanged) {
		learnThickness();
	}
}

bool FeatherGeomParam::isChanged() const
{ return _changed; }

int FeatherGeomParam::numSegments() const
{ return 3; }

const int & FeatherGeomParam::numFeatherOnSegment(int i) const
{ return _numFPerSeg[i]; }

const float * FeatherGeomParam::xOnSegment(int i) const
{ return _xsPerSeg[i]; }

void FeatherGeomParam::learnChord()
{
	float vx[4] = {0.01f, .33f, .67f, .99f};
	for(int i=0;i<4;++i) {
		m_chordInterp->setObservationi(i, &vx[i], &_chordLength[i]);
	}
	std::cout<<"\n learn chord"<<std::endl;
	
	if(!m_chordInterp->learn() ) {
		std::cout<<"FeatherGeomParam learnChord chord interpolate failed to learn";
	}
}

float FeatherGeomParam::predictChord(const float * x)
{
	m_chordInterp->predict(x);
	return *m_chordInterp->predictedY().column(0);
}

void FeatherGeomParam::learnThickness()
{
	float vx[4] = {0.01f, .33f, .67f, .99f};
	for(int i=0;i<4;++i) {
		m_thicknessInterp->setObservationi(i, &vx[i], &_thickness[i]);
	}
	std::cout<<"\n learn thickness"<<std::endl;
	
	if(!m_thicknessInterp->learn() ) {
		std::cout<<"FeatherGeomParam learnThickness thickness interpolate failed to learn";
	}
}

float FeatherGeomParam::predictThickness(const float * x)
{
	m_thicknessInterp->predict(x);
	return *m_thicknessInterp->predictedY().column(0);
}

const float & FeatherGeomParam::longestChord() const
{ return _longestChord; }
