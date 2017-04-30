/*
 *  BoxSampleProfile.h
 *  
 *
 *  Created by jian zhang on 3/17/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_IMG_BOX_SAMPLE_PROF_H
#define APH_IMG_BOX_SAMPLE_PROF_H

namespace aphid {

namespace img {

template<typename T>
struct BoxSampleProfile {
/// mult-scale sampling
	int _loLevel;
	int _hiLevel;
	float _mixing;
/// horizontal [0.0,1.0]
	float _uCoord;
/// vertical [0.0,1.0]
	float _vCoord;
	int _channel;
	T _defaultValue;
	T _box[4];

	bool isTexcoordOutofRange()
	{
		if(_uCoord < 0.f || _uCoord > 1.f) {
			return true;
		}
		if(_vCoord < 0.f || _vCoord > 1.f) {
			return true;
		}
		return false;
	}
	
};

}

}
#endif