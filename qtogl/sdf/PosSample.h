/*
 *  PosSample.h
 *  
 *
 *  Created by jian zhang on 2/15/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef POS_SAMPLE_H
#define POS_SAMPLE_H

#include <math/BoundingBox.h>
#include <geom/ConvexShape.h>

using namespace aphid;

struct PosSample {
	
	Vector3F _pos;
/// space-filling curve code
	int _key;
	Vector3F _nml;
	float _r;
	
	BoundingBox calculateBBox() const {
		return BoundingBox(_pos.x - _r, _pos.y - _r, _pos.z - _r,
						_pos.x + _r, _pos.y + _r, _pos.z + _r);
	}
	
	static std::string GetTypeStr() {
		return "possamp";
	}
	
	template<typename T>
	void closestToPoint(T * result) const
	{
		Vector3F tv = _pos - result->_toPoint;
		float d = tv.length();// - _r;
		if(d > Absolute<float>(result->_distance) ) {
			return;
		}
		
		tv.normalize();
		
		if(_nml.dot(tv) > -.1f ) {
			d = -d;
		}
		
		result->_distance = d;
		result->_hasResult = true;
		result->_hitPoint = _pos;
		result->_hitNormal = _nml;
		
	}
	
};

struct SampleInterp {
	
	bool reject(const PosSample& asmp ) const {
		return false;
	}
	
	void interpolate(PosSample& asmp,
				const float* coord,
				const cvx::Triangle* g) const {
	
	}
	
};


#endif
