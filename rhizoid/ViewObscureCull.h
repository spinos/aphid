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
#include <kd/KdEngine.h>

namespace aphid {

class ViewObscureCull : public ViewCull {

	IntersectionContext m_intersectCtx;
	
public:
	ViewObscureCull();
	virtual ~ViewObscureCull();
	
protected:
	
	template <typename T, typename Tr>
	bool cullByDepth(const Vector3F & pnt, const float & threshold,
					float & cameraZ,
					Tr * obscurer);
private:

};

template <typename T, typename Tr>
bool ViewObscureCull::cullByDepth(const Vector3F & pnt, const float & threshold,
					float & cameraZ,
					Tr * obscurer)
{
	cameraZ = eyePosition().distanceTo(pnt);
	if(!obscurer) return false;
	Ray incident(eyePosition(), pnt );
	m_intersectCtx.reset(incident);
	
	KdEngine engine;
	engine.intersect<T>(obscurer, &m_intersectCtx );
	
	if(!m_intersectCtx.m_success) return false;
	return ( cameraZ - eyePosition().distanceTo(m_intersectCtx.m_hitP) ) > threshold;
}

}