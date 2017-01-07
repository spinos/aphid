/*
 *  Geom1LineParam.cpp
 *  cinchona
 *
 *  Created by jian zhang on 1/3/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "Geom1LineParam.h"
#include <math/Vector3F.h>
#include <gpr/GPInterpolate.h>

using namespace aphid;

Geom1LineParam::Geom1LineParam(int nobs) 
{
	m_numObservations = nobs;
	linspace<float>(m_xTrain, 0.01f, .99f, nobs);
	
	_numTotalF = 0;
	for(int i=0;i< numSegments();++i) {
		_numFPerSeg[i] = 0;
		_xsPerSeg[i] = NULL;
		_psPerSeg[i] = NULL;
	}
	for(int i=0;i < nobs;++i) {
		_chordLength[i] = -1.f;
		_thickness[i] = -1.f;
	}
	m_chordInterp = new gpr::GPInterpolate<float>();
	m_chordInterp->create(nobs,1,1);
	m_chordInterp->setFilterLength(.43f);
	m_thicknessInterp = new gpr::GPInterpolate<float>();
	m_thicknessInterp->create(nobs,1,1);
	m_thicknessInterp->setFilterLength(.43f);
	m_rotateOffsetZ = 0.f;
}

Geom1LineParam::~Geom1LineParam()
{
	for(int i=0;i < numSegments();++i) {
		if(_xsPerSeg[i]) delete[] _xsPerSeg[i];
		if(_psPerSeg[i]) delete[] _psPerSeg[i];
	}
	delete m_chordInterp;
	delete m_thicknessInterp;
	
}

void Geom1LineParam::set(const int * nps,
						const float * chords,
						const float * ts)
{
	_changed = false;
	_numTotalF = 0;
	for(int i=0;i < numSegments();++i) {
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
	for(int i=0;i < m_numObservations;++i) {
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
	for(int i=0;i < m_numObservations;++i) {
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

bool Geom1LineParam::isChanged() const
{ return _changed; }

int Geom1LineParam::numSegments() const
{ return m_numObservations - 1; }

const int & Geom1LineParam::numFeatherOnSegment(int i) const
{ return _numFPerSeg[i]; }

const float * Geom1LineParam::xOnSegment(int i) const
{ return _xsPerSeg[i]; }

void Geom1LineParam::learnChord()
{
	for(int i=0;i < m_numObservations;++i) {
		m_chordInterp->setObservationi(i, &m_xTrain[i], &_chordLength[i]);
	}

	if(!m_chordInterp->learn() ) {
		std::cout<<"\n ERROR Geom1LineParam learnChord chord interpolate failed to learn";
	}
}

float Geom1LineParam::predictChord(const float * x)
{
	m_chordInterp->predict(x);
	return *m_chordInterp->predictedY().column(0);
}

void Geom1LineParam::learnThickness()
{
	for(int i=0;i < m_numObservations;++i) {
		m_thicknessInterp->setObservationi(i, &m_xTrain[i], &_thickness[i]);
	}

	if(!m_thicknessInterp->learn() ) {
		std::cout<<"\n ERROR Geom1LineParam learnThickness thickness interpolate failed to learn";
	}
}

float Geom1LineParam::predictThickness(const float * x)
{
	m_thicknessInterp->predict(x);
	return *m_thicknessInterp->predictedY().column(0);
}

const float & Geom1LineParam::longestChord() const
{ return _longestChord; }

const int Geom1LineParam::numGeoms() const
{
	int n = 0;
	for(int i=0;i < numSegments();++i) {
	    n += numFeatherOnSegment(i);
	}
	return n;
}

void Geom1LineParam::setRotateOffsetZ(float x)
{ m_rotateOffsetZ = x; }
	
const float & Geom1LineParam::rotateOffsetZ() const
{ return m_rotateOffsetZ; }
