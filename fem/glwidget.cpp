#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include <SolverThread.h>
#include <FEMTetrahedronMesh.h>
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
    qDebug()<<"glview";
	perspCamera()->setFarClipPlane(2000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(2000.f);
	orthoCamera()->setNearClipPlane(1.f);
	
	m_solver = new SolverThread(this);
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
    m_solver->initOnDevice();
    connect(internalTimer(), SIGNAL(timeout()), m_solver, SLOT(simulate()));
	connect(m_solver, SIGNAL(doneStep()), this, SLOT(update()));
	disconnect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
	FEMTetrahedronMesh * mesh = m_solver->mesh();
    glColor3f(0.5f, 0.6f, .44f);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	
	drawMesh(mesh);
	
	glColor3f(0.1f, 0.14f, .5f);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
	drawMesh(mesh);
}

void GLWidget::drawMesh(FEMTetrahedronMesh * mesh)
{
	unsigned nt = mesh->numTetrahedra();
    FEMTetrahedronMesh::Tetrahedron * t = mesh->tetrahedra();
	Vector3F * p = mesh->X();
	Vector3F q;
	glBegin(GL_TRIANGLES);
	unsigned i, j;
	for(i = 0; i < nt; i++) {
	    FEMTetrahedronMesh::Tetrahedron & tet = t[i];
		
		for(j=0; j< 12; j++) {
            q = p[ tet.indices[ TetrahedronToTriangleVertex[j] ] ];
            glVertex3fv((GLfloat *)&q);
        }
	}
	glEnd();
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
//! [10]

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	Base3DView::keyPressEvent(e);
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base3DView::keyReleaseEvent(event);
}

void GLWidget::simulate()
{
    m_solver->stepPhysics(1.f/60.f);
    update();
}
