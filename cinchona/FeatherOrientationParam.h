/*
 *  FeatherOrientationParam.h
 *  cinchona
 *
 *  Created by jian zhang on 1/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef FEATHER_ORIENTATION_PARAM_H
#define FEATHER_ORIENTATION_PARAM_H

namespace aphid {

class Vector3F;
class Matrix33F;

namespace gpr {
template<typename T>
class GPInterpolate;

}

}

class FeatherOrientationParam {

	aphid::Vector3F * m_vecs;
	aphid::Matrix33F * m_rots;
	aphid::gpr::GPInterpolate<float > * m_sideInterp;
	aphid::gpr::GPInterpolate<float > * m_upInterp;
/// 0 flying
/// 1:2 upper covert
/// 3:4 lower covert
	float m_rzOffset[5];
	float m_yawNoiseWeight;
	bool m_changed;
	
public:
	FeatherOrientationParam();
	virtual ~FeatherOrientationParam();
	
	void set(const aphid::Matrix33F * mats);
/// rzs[4] 0:1 upper 2:3 lower covert offset
	void set(const aphid::Matrix33F * mats, 
			const float * rzs,
			const float * yawNoiseWeight);
		
	bool isChanged() const;
	
	const aphid::Matrix33F & rotation(int i) const;
	
	void predictRotation(aphid::Matrix33F & dst,
						const float * x);
/// plus i-th line offset
	void predictLineRotation(aphid::Matrix33F & dst,
						const int & iline,
						const float * x);
						
	const float * yawNoise() const;
	
protected:
	aphid::Vector3F * rotationSideR(int i);
	aphid::Vector3F * rotationUpR(int i);
	void learnOrientation();
	aphid::gpr::GPInterpolate<float > * sideInterp();
	aphid::gpr::GPInterpolate<float > * upInterp();
	
private:
	
};
#endif