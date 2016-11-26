#include <QtGui>
#include <QtOpenGL>
#include <GeodesicSphereMesh.h>
#include "instwidget.h"
#include <GeoDrawer.h>
#include <GlslInstancer.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{
    m_sphere = new TriangleGeodesicSphere(10);
    m_instancer = new GlslInstancer;
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
    std::string diaglog;
    m_instancer->diagnose(diaglog);
    std::cout<<diaglog;
    std::cout.flush();
    
}

void GLWidget::clientDraw()
{
    getDrawer()->m_paintProfile.apply();
	getDrawer()->setColor(.75f, .85f, .7f);

	m_instancer->programBegin();
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glNormalPointer(GL_FLOAT, 0, (GLfloat*)m_sphere->vertexNormals());
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)m_sphere->points());
	glDrawElements(GL_TRIANGLES, m_sphere->numIndices(), GL_UNSIGNED_INT, m_sphere->indices());

	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	m_instancer->programEnd();
	
}


