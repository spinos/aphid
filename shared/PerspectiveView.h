/*
 *  PerspectiveView.h
 *  aphid
 *
 *  Created by jian zhang on 8/3/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include "PerspectiveCamera.h"
#include <geom/ConvexShape.h>

namespace aphid {

class PerspectiveView {

	cvx::Frustum m_frustum;
	const PerspectiveCamera * m_camera;
	float m_frameBase;
	
public:
	PerspectiveView(const PerspectiveCamera * camera);
	virtual ~PerspectiveView();
	
	void update();
	
	const cvx::Frustum * frustum() const;
	
	bool isBoxVisible(const BoundingBox & b) const;
	float lodBox(const BoundingBox & b) const;
	
protected:

private:

};

}