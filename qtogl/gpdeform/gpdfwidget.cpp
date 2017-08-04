#include <QtGui>
#include <GeoDrawer.h>
#include <QtOpenGL>
#include "gpdfwidget.h"
#include "../cactus.h"
#include <gpr/GPInterpolate.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{	
	m_interp = new gpr::GPInterpolate<float>();
	m_interp->create(sCactusNumObservations, 2, sCactusNumVertices*3);
	for(int i=0;i<sCactusNumObservations;++i) {
		m_interp->setObservationi(i, sCactusXValues[i], sCactusMeshVertices[i]);
	}
	
	if(!m_interp->learn() ) {
		qDebug()<<"gp interpolate failed to learn";
	}
	
	float xtest[2] = {0.f, 0.f};
	m_interp->predict(xtest);
	
}

GLWidget::~GLWidget()
{
	delete m_interp;
}

void GLWidget::clientInit()
{}

void GLWidget::clientDraw()
{
    getDrawer()->m_wireProfile.apply();

	getDrawer()->setColor(0.f, .0f, .55f);

	glEnableClientState(GL_VERTEX_ARRAY);
		    
	for(int i=1;i<sCactusNumObservations;++i) {
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)sCactusMeshVertices[i]);
		glDrawArrays(GL_POINTS, 0, sCactusNumVertices );
		//glDrawElements(GL_TRIANGLES, sCactusNumTriangleIndices, GL_UNSIGNED_INT, sCactusMeshTriangleIndices );
		
	}
	
	const float * ytest = m_interp->predictedY().column(0);
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)ytest);
	glDrawElements(GL_TRIANGLES, sCactusNumTriangleIndices, GL_UNSIGNED_INT, sCactusMeshTriangleIndices );
	
	glDisableClientState(GL_VERTEX_ARRAY);
	
	getDrawer()->m_markerProfile.apply();
	getDrawer()->setColor(0.55f, .55f, .45f);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)ytest);
	glDrawElements(GL_TRIANGLES, sCactusNumTriangleIndices, GL_UNSIGNED_INT, sCactusMeshTriangleIndices );
			
	glDisableClientState(GL_VERTEX_ARRAY);
	
}

void GLWidget::recvXValue(QPointF vx)
{ 
	float xtest[2] = {vx.x(), vx.y()};
	m_interp->predict(xtest);
	update();
}
