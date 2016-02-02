/*
 *  ViewDepthCull.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <ViewCull.h>
class DepthCull;
class ViewDepthCull : public ViewCull {

	DepthCull * m_depthCuller;
	
public:
	ViewDepthCull();
	virtual ~ViewDepthCull();
	
protected:
/// must with gl context
	void initDepthCull();
	
	void diagnoseDepthCull();
	bool isDepthCullDiagnosed() const;
	void initDepthCullFBO();
	DepthCull * depthCuller();
	
	bool cullByDepth(const Vector3F & pnt, const float & threshold) const;
	
private:

};