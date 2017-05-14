#include <QtGui>
#include <QtOpenGL>
#include "glwidget.h"
#include <topo/ConvexHullGen.h>
#include <geom/ATriangleMesh.h>
#include <GeoDrawer.h>
#include <cmath>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{
	ConvexHullGen _dynamics;
	//_dynamics.addSample(Vector3F(0.f, 32.f, 0.f));
	int nv = 999;
	Vector3F vp;
	for(int i = 0; i < nv; i++) {
		float r = ((float)(rand() % 24091)) / 24091.f * 11.1f + 12.f;
		float phi = ((float)(rand() % 25391)) / 25391.f * 2.f * 3.14f;
		float theta = ((float)(rand() % 24331)) / 24331.f * 3.14f;
		
		vp.x = sin(theta) * cos(phi) * r * 2.194f;
		vp.y = sin(theta) * sin(phi) * r + 39.f;
		vp.z = cos(theta) * r * 1.181f;
		
		_dynamics.addSample(vp);
	}
	
	_dynamics.processHull();
	
	m_tri = new ATriangleMesh;
	_dynamics.extractMesh(m_tri);
	
	std::cout<<"\n convex hull n horizon "<<_dynamics.getNumFace();
	std::cout.flush();
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{}

void GLWidget::clientDraw()
{	
	getDrawer()->m_surfaceProfile.apply();
	//getDrawer()->m_wireProfile.apply();
	//getDrawer()->setColor(1,.7,0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	
	const unsigned nind = m_tri->numIndices();
	const unsigned * inds = m_tri->indices();
	const Vector3F * pos = m_tri->points();
	const Vector3F * nml = m_tri->vertexNormals();
	
	glNormalPointer(GL_FLOAT, 0, (GLfloat*)nml );
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)pos );
	glDrawElements(GL_TRIANGLES, nind, GL_UNSIGNED_INT, inds);

	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}
