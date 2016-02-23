/*
 *  AAnimationCurve.cpp
 *  aphid
 *
 *  Created by jian zhang on 8/27/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "AAnimationCurve.h"

namespace aphid {

AAnimationCurve::AAnimationCurve() 
{ m_curveType = TUnknown; }

AAnimationCurve::~AAnimationCurve() 
{ m_keys.clear(); }

void AAnimationCurve::setCurveType(CurveType x)
{ m_curveType = x; }

AAnimationCurve::CurveType AAnimationCurve::curveType() const
{ return m_curveType; }
	
void AAnimationCurve::addKey(const AAnimationKey & x)
{ m_keys.push_back(x); }

unsigned AAnimationCurve::numKeys() const
{ return m_keys.size(); }

AAnimationKey AAnimationCurve::key(unsigned i) const
{ return m_keys[i]; }

}
//:~
