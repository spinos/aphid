/*
 *  GProfile.h
 *  mallard
 *
 *  Created by jian zhang on 9/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <GProfile.h>

GProfile::GProfile() : m_lighting(false),
		m_depthTest(false),
		m_wired(false),
		m_culled(false),
		m_material(0) {}
		
GProfile::GProfile(bool lighting, bool depthTest, bool wired, bool culled) 
{
	m_lighting = lighting;
	m_depthTest = depthTest; 
	m_wired = wired; 
	m_culled = culled;
	m_material = 0;
}
	
void GProfile::apply() const
{
	if(m_lighting) glEnable(GL_LIGHTING);
	else glDisable(GL_LIGHTING);
	
	if(m_depthTest) glEnable(GL_DEPTH_TEST);
	else glDisable(GL_DEPTH_TEST);
	
	if(m_wired) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	
	if(m_culled) glEnable(GL_CULL_FACE);
	else glDisable(GL_CULL_FACE);
	
	if(m_material) m_material->apply();
}