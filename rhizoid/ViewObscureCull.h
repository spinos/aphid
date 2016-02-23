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
	
private:

};

}