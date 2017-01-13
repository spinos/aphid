#include <QtGui>
#include <QtOpenGL>
#include "widget.h"
#include <GeoDrawer.h>
#include <ogl/DrawCircle.h>
#include <ogl/RotationHandle.h>
#include <BaseCamera.h>
#include <ogl/DrawBox.h>
#include <ogl/DrawDop.h>
#include <math/AOrientedBox.h>
#include <ConvexShape.h>
#include <sdb/VectorArray.h>
#include <kd/KdEngine.h>
#include "../cactus.h"

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{ 
	usePerspCamera(); 
	m_space.translate(1,1,1);
	m_roth = new RotationHandle(&m_space);
	m_triangles = new sdb::VectorArray<cvx::Triangle>();
	
/// prepare kd tree
	BoundingBox gridBox;
	KdEngine eng;
	eng.buildSource<cvx::Triangle, 3 >(m_triangles, gridBox,
									sCactusMeshVertices[0],
									sCactusNumTriangleIndices,
									sCactusMeshTriangleIndices);
									
	std::cout<<"\n kd tree source bbox"<<gridBox;
	
	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 64;
	
	TreeTyp * m_tree = new TreeTyp;
	
	eng.buildTree<cvx::Triangle, KdNode4, 4>(m_tree, m_triangles, gridBox, &bf);
	
	std::cout.flush();
	
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{}

void GLWidget::clientDraw()
{
	getDrawer()->m_markerProfile.apply();

	getDrawer()->setColor(0.f, .35f, .45f);

	getDrawer()->m_surfaceProfile.apply();
	
	getDrawer()->m_wireProfile.apply();
	
	glEnableClientState(GL_VERTEX_ARRAY);
	
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)sCactusMeshVertices[0] );
	glDrawElements(GL_TRIANGLES, sCactusNumTriangleIndices, GL_UNSIGNED_INT, sCactusMeshTriangleIndices );
	
	glDisableClientState(GL_VERTEX_ARRAY);
}

void GLWidget::clientSelect(QMouseEvent *event)
{
	m_roth->begin(getIncidentRay() );
	update();
}

void GLWidget::clientDeselect(QMouseEvent *event)
{
	m_roth->end();
	update();
}

void GLWidget::clientMouseInput(QMouseEvent *event)
{
	m_roth->rotate(getIncidentRay() );
	
	update();
}
	