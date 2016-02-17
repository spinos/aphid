/*
 *  depthCull.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 5/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "DepthCull.h"
#include <ATriangleMesh.h>

DepthCull::DepthCull() {}
DepthCull::~DepthCull() {}

void DepthCull::drawFrameBuffer(const std::vector<ATriangleMesh *> & meshes)
{		
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	
	programBegin();
	
	glPushMatrix();
	glMultMatrixd(fLocalSpace);
	glColor3f(1,1,1);
	glBegin(GL_TRIANGLES);
	
	std::vector<ATriangleMesh *>::const_iterator it = meshes.begin();
	for(;it!=meshes.end();++it) {
		ATriangleMesh * mesh = *it;
		const unsigned nt = mesh->numTriangles();
		Vector3F * p = mesh->points();
		for(unsigned i=0; i<nt; ++i) {
			unsigned * tri = mesh->triangleIndices(i);
			glVertex3fv((const GLfloat *)&p[tri[0]]);
			glVertex3fv((const GLfloat *)&p[tri[1]]);
			glVertex3fv((const GLfloat *)&p[tri[2]]);
		}
	}
	glEnd();
	glPopMatrix();
	
	programEnd();
}

void DepthCull::setLocalSpace(double* m)
{
	fLocalSpace = m;
}

const char* DepthCull::vertexProgramSource() const
{
	return "varying vec3 PCAM;"
"void main()"
"{"
"		gl_Position = ftransform();"
"		gl_FrontColor = gl_Color;"
"	PCAM = vec3 (gl_ModelViewMatrix * gl_Vertex);"
"}";
}

const char* DepthCull::fragmentProgramSource() const
{
	return "varying vec3 PCAM;"
"void main()"
"{"
"	float d = -PCAM.z; "
"		gl_FragColor = vec4 (d, d/10000.0, d/10000.0, 1.0);"
"}";
}

float DepthCull::getBufferDepth(float s, float t) const
{
	if(!hasFBO()) return -1.f;
		
	const int coordx = s * frameBufferWidth();
	const int coordy = t * frameBufferHeight();
	return pixels()[(coordy * frameBufferWidth() + coordx) * 4];
	
	//if(buffed < 0.1f) return 0;
	//return depth > (buffed + offset);
}

