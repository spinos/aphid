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
	const int numGeoms() const;
	
	const int & numFeatherOnSegment(int i) const;
/// i-th segment
	const float * xOnSegment(int i) const;
	
	float predictChord(const float * x);
	float predictThickness(const float * x);
	const float & longestChord() const;
	void calculateX(float * xs) const;
	
private:
	void learnChord();
	void learnThickness();
	
};
	
#endif