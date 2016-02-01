/*
 *  depthCut.cpp
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
	programBegin();
	
	glPushMatrix();
	glMultMatrixd(fLocalSpace);
	
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
"		gl_FragColor = vec4 (d, d, d, 1.0);"
"}";
}

char DepthCull::isCulled(float depth, int x, int y, int w, int h, float offset)
{		
	if(!hasFBO()) return 0;

	if(x < 0 || y < 0 || x >= w || y >=h)
		return 0;
		
	const int coordx = (float)x/(float)w * (frameBufferWidth());
	const int coordy = (float)y/(float)h * (frameBufferHeight());
	const float buffed = fPixels[(coordy * frameBufferWidth() + coordx) * 4];
	
	if(buffed < 0.1f) return 0;

	//MGlobal::displayInfo(MString("d:")+depth+" "+buffed+" "+offset);

	return depth > (buffed + offset);
}

