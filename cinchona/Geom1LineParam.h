/*
 *  Geom1LineParam.h
 *  cinchona
 *
 *  Created by jian zhang on 1/3/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GEOM_1_LINE_PARAM_H
#define GEOM_1_LINE_PARAM_H

#include <iostream>
#include <math/linspace.h>

namespace aphid {

class Vector3F;

namespace gpr {
template<typename T>
class GPInterpolate;

}

}

class Geom1LineParam {

#define NUM_OBS_MAX 8
	int m_numObservations;
	float m_xTrain[NUM_OBS_MAX];
	int _numFPerSeg[NUM_OBS_MAX];
	int _numTotalF;
	float * _xsPerSeg[NUM_OBS_MAX];
	float _chordLength[NUM_OBS_MAX];
	float _thickness[NUM_OBS_MAX];
	float _longestChord;
	float m_rotateOffsetZ;
	aphid::Vector3F * _psPerSeg[NUM_OBS_MAX]; 
	aphid::gpr::GPInterpolate<float > * m_chordInterp;
	aphid::gpr::GPInterpolate<float > * m_thicknessInterp;
	bool _changed;
		
public:
/// nbs number of observations
/// nbs - 1 segments
	Geom1LineParam(int nobs);
	~Geom1LineParam();	
	
	void set(const int * nps,
			const float * chords,
			const float * ts);
	
	bool isChanged() const;
	
	int numSegments() const;
	int numGeoms() const;
/// each segment minus 1
	int numGeomsM1() const;
	
	const int & numFeatherOnSegment(int i) const;
/// i-th segment
	const float * xOnSegment(int i) const;
	
	float predictChord(const float * x);
	float predictThickness(const float * x);
	const float & longestChord() const;
	
	template<typename T>
	void Geom1LineParam::calculateX(float * xs,
						const T * line) const
	{
		int nseg;
		float totalL;
		float * segLs;
		line->getLengths(segLs, nseg, totalL);
		
		for(int i=0;i < nseg;++i) {
			segLs[i] /= totalL;
			//std::cout<<" l["<<i<<"] = "<<segLs[i];
		}
		
		float segL = 0.f;
		float segH = segLs[0];
		int segI = 0;
		for(int i=0;i < nseg;++i) {
			const int & nf = numFeatherOnSegment(i);
			aphid::linspace<float>(&xs[segI], segL, segH, nf);
			
			segI += nf;
			segL = segH;
			if((i+1) < nseg ) {
				segH += segLs[i+1];
			}
		}
		
		delete[] segLs;
#if 0
		int ng = numGeoms();
		for(int i=0;i < ng;++i) {
			std::cout<<" x["<<i<<"] = "<<xs[i];
		}
#endif
	}
	
	void setRotateOffsetZ(float x);
	const float & rotateOffsetZ() const;
	
private:
	void learnChord();
	void learnThickness();
	
};
	
#endif