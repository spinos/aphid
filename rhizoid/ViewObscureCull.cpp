/*
 *  ViewObscureCull.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ViewObscureCull.h"
#include <KdTree.h>

namespace aphid {

ViewObscureCull::ViewObscureCull() {}
ViewObscureCull::~ViewObscureCull() {}

bool ViewObscureCull::cullByDepth(const Vector3F & pnt, const float & threshold,
					float & cameraZ,
					KdTree * obscurer)
{
	cameraZ = eyePosition().distanceTo(pnt);
	if(!obscurer) return false;
	Ray incident(eyePosition(), pnt );
	m_intersectCtx.reset(incident);
	obscurer->intersect(&m_intersectCtx );
	
	if(!m_intersectCtx.m_success) return false;
	return ( cameraZ - eyePosition().distanceTo(m_intersectCtx.m_hitP) ) > threshold;
}

}
//:~