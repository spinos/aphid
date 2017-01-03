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

}

class FeatherGeomParam {

	int _numFPerSeg[4];
	int _numTotalF;
	float * _xsPerSeg[4];
	aphid::Vector3F * _psPerSeg[4]; 
	bool _changed;
		
public:
	FeatherGeomParam();
	~FeatherGeomParam();	
	
	void set(const int * nps);
	
	bool isChanged() const;
	
	int numSegments() const;
	const int & numFeatherOnSegment(int i) const;
/// i-th segment
	const float * xOnSegment(int i) const;
	
};
	
#endif