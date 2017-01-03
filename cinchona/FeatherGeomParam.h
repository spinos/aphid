/*
 *  FeatherGeomParam.h
 *  cinchona
 *
 *  Created by jian zhang on 1/3/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef FEATHER_GEOM_PARAM_H
#define FEATHER_GEOM_PARAM_H

namespace aphid {

class Vector3F;

namespace gpr {
template<typename T>
class GPInterpolate;

}

}

class FeatherGeomParam {

	int _numFPerSeg[3];
	int _numTotalF;
	float * _xsPerSeg[3];
	float _chordLength[4];
	float _thickness[4];
	aphid::Vector3F * _psPerSeg[3]; 
	aphid::gpr::GPInterpolate<float > * m_chordInterp;
	aphid::gpr::GPInterpolate<float > * m_thicknessInterp;
	bool _changed;
		
public:
	FeatherGeomParam();
	~FeatherGeomParam();	
	
	void set(const int * nps,
			const float * chords,
			const float * ts);
	
	bool isChanged() const;
	
	int numSegments() const;
	const int & numFeatherOnSegment(int i) const;
/// i-th segment
	const float * xOnSegment(int i) const;
	
	float predictChord(const float * x);
	float predictThickness(const float * x);
	
private:
	void learnChord();
	void learnThickness();
	
};
	
#endif