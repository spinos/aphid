/*
 *  DrawSample.cpp
 *  
 *  as points
 *
 *  Created by jian zhang on 1/13/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawSample.h"
#include <gl_heads.h>

namespace aphid {

DrawSample::DrawSample()
{}

void DrawSample::begin(const Profile & prof)
{
	m_prof = prof;
	
	glPointSize(prof.m_pointSize);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	if(prof.m_hasNormal) {
		glEnableClientState(GL_NORMAL_ARRAY);
	}
}

void DrawSample::end() const
{
	if(m_prof.m_hasNormal) {
		glDisableClientState(GL_NORMAL_ARRAY);
	}
	glDisableClientState(GL_VERTEX_ARRAY);
}

void DrawSample::draw(const float * points,
				const float * normals,
				const int & count) const
{
	glNormalPointer(GL_FLOAT, m_prof.m_stride, normals );
	glVertexPointer(3, GL_FLOAT, m_prof.m_stride, points );
	glDrawArrays(GL_POINTS, 0, count);
}

}
