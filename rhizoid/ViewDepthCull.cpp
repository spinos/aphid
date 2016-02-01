/*
 *  ViewDepthCull.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ViewDepthCull.h"
#include "DepthCull.h"
#include <string>
#include <iostream>

ViewDepthCull::ViewDepthCull() : m_depthCuller(NULL) {}
ViewDepthCull::~ViewDepthCull() 
{
	if(m_depthCuller) delete m_depthCuller;
}

void ViewDepthCull::initDepthCull()
{
	if(!m_depthCuller) m_depthCuller = new DepthCull;
}

bool ViewDepthCull::isDepthCullDiagnosed() const
{ return m_depthCuller->isDiagnosed(); }

void ViewDepthCull::diagnoseDepthCull()
{ 
	std::string log;
	m_depthCuller->diagnose(log);
	std::cout<<"\n glsl diagnose log: "<<log;
}

void ViewDepthCull::initDepthCullFBO()
{
	if(!m_depthCuller->hasFBO()) {
		std::string log;
		m_depthCuller->initializeFBO(log);
		std::cout<<"\n fbo diagnose log: "<<log;
	}
}

DepthCull * ViewDepthCull::depthCuller()
{ return m_depthCuller; }

//:~