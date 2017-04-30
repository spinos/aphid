#ifndef AANIMATIONCURVE_H
#define AANIMATIONCURVE_H

/*
 *  AAnimationCurve.h
 *  aphid
 *
 *  Created by jian zhang on 8/27/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <deque>

namespace aphid {

class AAnimationKey {
public:
	float _key;
	float _value;
	float _inAngle;
	float _outAngle;
	float _inWeight;
	float _outWeight;
	int _inTangentType;
	int _outTangentType;
};

class AAnimationCurve {
public:
	enum CurveType {
		TUnknown = 0,
		TTL = 1,
		TTA = 2,
		TTU = 3
	};
	
	AAnimationCurve();
	virtual ~AAnimationCurve();
	
	void setCurveType(CurveType x);
	CurveType curveType() const;
	
	void addKey(const AAnimationKey & x);
	unsigned numKeys() const;
	AAnimationKey key(unsigned i) const;
	
protected:

private:
	std::deque<AAnimationKey> m_keys;
	CurveType m_curveType;
};

}
#endif        //  #ifndef AANIMATIONCURVE_H
