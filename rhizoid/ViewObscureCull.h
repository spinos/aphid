/*
 *  ViewObscureCull.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <ViewCull.h>
#include <IntersectionContext.h>

namespace aphid {

class KdTree;
class ViewObscureCull : public ViewCull {

	IntersectionContext m_intersectCtx;
	
public:
	ViewObscureCull();
	virtual ~ViewObscureCull();
	
protected:

	bool cullByDepth(const Vector3F & pnt, const float & threshold,
					float & cameraZ,
					KdTree * obscurer);
	
	template <typename T>
	bool cullByDepth(const Vector3F & pnt, const float & threshold,
					float & cameraZ,
					T * obscurer);
private:

};

template <typename T>
bool ViewObscureCull::cullByDepth(const Vector3F & pnt, const float & threshold,
					float & cameraZ,
					T * obscurer)
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