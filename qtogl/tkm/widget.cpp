/*
 *  widget.cpp
 *  Kmean clustering Test
 *
 */
#include <GeoDrawer.h>
#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include <PerspectiveView.h>
#include "widget.h"
#include <math/miscfuncs.h>
#include <math/kmean.h>
#include <ogl/DrawCircle.h>

using namespace aphid;

#define NUM_PNT 200

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{	
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
	
	m_pos = new Vector3F[NUM_PNT];
	m_nml = new Vector3F[NUM_PNT];
	
	for(int i=0;i<NUM_PNT;++i) {
		m_pos[i].set(RandomF01() * 32.f, RandomF01() * 32.f, 0.f);
		m_nml[i].set(RandomF01(), RandomF01(), 0.f);
	}
	
	int n = NUM_PNT;
	int k = 10;
	
	const int d = 6;
/// to kmean data
	DenseMatrix<float> data(n, d);
	for(int i=0;i<n;++i) {
		const Vector3F & srcp = m_pos[i];
		const Vector3F & srcn = m_nml[i];
		data.column(0)[i] = srcp.x;
		data.column(1)[i] = srcp.y;
		data.column(2)[i] = srcp.z;
		data.column(3)[i] = srcn.x * 2.5f;
		data.column(4)[i] = srcn.y * 2.5f;
		data.column(5)[i] = srcn.z * 2.5f;
	}
	
	m_cluster = new KMeansClustering2<float>();
	m_cluster->setKND(k, n, d);
	if(!m_cluster->compute(data) ) {
		std::cout<<"\n kmean failed ";
		return;
	}
	
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
    
}

void GLWidget::clientDraw()
{
	getDrawer()->m_wireProfile.apply();
	getDrawer()->setColor(.125f, .125f, .5f);

    glBegin(GL_LINES);
	for(int i=0;i<NUM_PNT;++i) {
		int g = m_cluster->groupIndices()[i];
		getDrawer()->setGroupColorLight(g);
		
		Vector3F pe = m_pos[i] + m_nml[i];
	
		glVertex3fv((const GLfloat *)&m_pos[i] );
		glVertex3fv((const GLfloat *)&pe );
		
	}
	glEnd();
	
	Matrix44F mf;
	float mat[16];
	mf.glMatrix(mat);
	DenseVector<float> centr;
	
	DrawCircle dc;
	for(int i=0;i<m_cluster->K();++i) {
		m_cluster->getGroupCentroid(centr, i);
		
		getDrawer()->setGroupColorLight(i);
		mat[12] = centr[0];
		mat[13] = centr[1];
		mat[14] = centr[2];
		
		dc.drawZCircle(mat);
	}
	
    getDrawer()->m_markerProfile.apply();
    getDrawer()->setColor(.05f, .5f, .15f);
	
	getDrawer()->m_surfaceProfile.apply();
	
}

void GLWidget::clientSelect(Vector3F & origin, Vector3F & ray, Vector3F & hit)
{
}
//! [9]

void GLWidget::clientDeselect()
{
}

//! [10]
void GLWidget::clientMouseInput(Vector3F & stir)
{
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	switch (e->key()) {
		case Qt::Key_M:
			break;
		case Qt::Key_N:
			break;
		default:
			break;
	}
	Base3DView::keyPressEvent(e);
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base3DView::keyReleaseEvent(event);
}
	
