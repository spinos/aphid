/*
 *  FeatherDeformParam.h
 *  cinchona
 *
 *  Created by jian zhang on 1/5/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef FEATHER_DEFORM_PARAM_H
#define FEATHER_DEFORM_PARAM_H

namespace aphid {

class Vector3F;
class Matrix33F;

namespace gpr {
template<typename T>
class GPInterpolate;

}

}

class FeatherDeformParam {

	aphid::Vector3F * m_vecs;
	aphid::Matrix33F * m_rots;
	aphid::gpr::GPInterpolate<float > * m_sideInterp;
	aphid::gpr::GPInterpolate<float > * m_upInterp;
	
	bool m_changed;
	
public:
	FeatherDeformParam();
	~FeatherDeformParam();
	
	void set(const aphid::Matrix33F * mats);
			
	bool isChanged() const;
	
	const aphid::Matrix33F & rotation(int i) const;
	
	void predictRotation(aphid::Matrix33F & dst,
						const float * x);
	
protected:
	aphid::Vector3F * rotationSideR(int i);
	aphid::Vector3F * rotationUpR(int i);
	
private:
	void learnOrientation();
	
};
#endif
