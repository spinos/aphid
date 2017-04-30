/*
 *  DrawInstance.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 12/31/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawInstance.h"
#include <ogl/GlslInstancer.h>

namespace aphid {

GlslLegacyInstancer * DrawInstance::m_instancer = new GlslLegacyInstancer;
GlslLegacyFlatInstancer * DrawInstance::m_wireInstancer = new GlslLegacyFlatInstancer;

DrawInstance::DrawInstance()
{}

DrawInstance::~DrawInstance()
{}

bool DrawInstance::isGlslReady() const
{
	return (GLSLBase::isDiagnosed()
			&& m_instancer->hasShaders()
			&& m_wireInstancer->hasShaders() );
}

bool DrawInstance::prepareGlsl()
{
	std::string log;
	bool stat = GLSLBase::diagnose(log);
	std::cout<<"\n "<<log;
	if(!stat) {
		return stat;
	}
	
	if(!m_instancer->hasShaders() ) {
		stat = m_instancer->initializeShaders(log);
		std::cout<<"\n "<<log;
	}
	
	if(!stat) {
		return stat;
	}
		
	if(!m_wireInstancer->hasShaders() ) {
		stat = m_wireInstancer->initializeShaders(log);
		std::cout<<"\n "<<log;
	}
	
	if(!stat) {
		return stat;
	}
	
	return true;
}

}