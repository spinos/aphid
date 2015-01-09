#include <QtGui>

#include <gl_heads.h>
#include "glwidget.h"

#include <KdTreeDrawer.h>
#include <CUDABuffer.h>
#include <BvhSolver.h>
#include "bvh_common.h"

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_solver = new BvhSolver;
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
	m_solver->init();
	//m_cvs->create(m_curve->numVertices() * 12);
	//m_cvs->hostToDevice(m_curve->m_cvs, m_curve->numVertices() * 12);
	//m_program->run(m_vertexBuffer, m_cvs, m_curve);
	connect(internalTimer(), SIGNAL(timeout()), m_solver, SLOT(simulate()));
	connect(m_solver, SIGNAL(doneStep()), this, SLOT(update()));
	// connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
    glEnableClientState(GL_VERTEX_ARRAY);

	glVertexPointer(4, GL_FLOAT, 0, (GLfloat*)m_solver->displayVertex());
	glDrawElements(GL_TRIANGLES, m_solver->getNumTriangleFaceVertices(), GL_UNSIGNED_INT, m_solver->getIndices());

	glDisableClientState(GL_VERTEX_ARRAY);
	
	// showEdgeContacts();
	showAabbs();
	m_solver->setAlpha((float)elapsedTime()/300.f);
	//qDebug()<<"drawn in "<<deltaTime();
}

void GLWidget::showEdgeContacts()
{
    glPolygonMode(GL_FRONT, GL_FILL);
    glPolygonMode(GL_BACK, GL_LINE);
    
    float * dsyV = m_solver->displayVertex();
	EdgeContact * ec = m_solver->edgeContacts();
	unsigned ne = m_solver->numEdges();
	unsigned a, b, c, d;
	const float h = 0.2f;
	const unsigned maxI = m_solver->numVertices();
	float * p;
	glBegin(GL_TRIANGLES);
	for(unsigned i=0; i < ne; i++) {
	    EdgeContact & ae = ec[i];
	    a = ae.v[0];
	    b = ae.v[1];
	    c = ae.v[2];
	    d = ae.v[3];
	    
	    if(c < maxI && d < maxI) {
	        p = &dsyV[a * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        p = &dsyV[b * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        p = &dsyV[c * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        
	        p = &dsyV[b * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        p = &dsyV[a * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        p = &dsyV[d * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	    }
	    else if(c < maxI) {
	        p = &dsyV[a * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        p = &dsyV[b * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        p = &dsyV[c * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	    }
	    else if(d < maxI) {
	        p = &dsyV[a * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        p = &dsyV[b * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	        p = &dsyV[d * 4];
	        glVertex3f(p[0], p[1] + h, p[2]);
	    }
	}
	glEnd();
}
void GLWidget::showAabbs()
{
	Aabb ab = m_solver->combinedAabb();
	GeoDrawer * dr = getDrawer();
    BoundingBox bb; 
	bb.setMin(ab.low.x, ab.low.y, ab.low.z);
	bb.setMax(ab.high.x, ab.high.y, ab.high.z);
	glColor3f(0.f, 0.5f, 0.2f);
    dr->boundingBox(bb);
#ifdef BVHSOLVER_DBG_DRAW
    Aabb * boxes = m_solver->displayAabbs();
    unsigned ne = m_solver->numEdges();
    for(unsigned i=0; i < ne; i++) {
        ab = boxes[i];
        BoundingBox bb; 
        bb.setMin(ab.low.x, ab.low.y, ab.low.z);
        bb.setMax(ab.high.x, ab.high.y, ab.high.z);
        dr->boundingBox(bb);
    }
	
	boxes = m_solver->displayCombinedAabb();
    for(unsigned i=0; i < 8; i++) {
        ab = boxes[3];
        BoundingBox bb; 
        bb.setMin(ab.low.x, ab.low.y, ab.low.z);
        bb.setMax(ab.high.x, ab.high.y, ab.high.z);
        dr->boundingBox(bb);
    }
#endif
}

void GLWidget::clientSelect(QMouseEvent */*event*/)
{
	setUpdatesEnabled(false);
	
	setUpdatesEnabled(true);
}

void GLWidget::clientDeselect(QMouseEvent */*event*/) 
{
	setUpdatesEnabled(false);
	
	setUpdatesEnabled(true);
}

void GLWidget::clientMouseInput(QMouseEvent */*event*/)
{
	setUpdatesEnabled(false);
	
	setUpdatesEnabled(true);
}

