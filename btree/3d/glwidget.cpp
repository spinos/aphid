#include <QtGui>
#include <QtOpenGL>
#include <Types.h>
#include "glwidget.h"
#include <cmath>
#include <KdTreeDrawer.h>
#include <Sequence.h>
#include <Ordered.h>
#define NUMVERTEX 190000

struct X {
	int a, b;
};
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
    int i;
	float w = 29.3f, h = 3.f, d = 13.f;
    m_pool = new PNPrefW[NUMVERTEX];
	for(i = 0; i < NUMVERTEX; i++) {
        Vector3F t;
		if(i > NUMVERTEX/2) {
			w = 5.f;
			h = 24.3f;
			d = 3.f;
		}
        t.x = (float(rand()%694) / 694.f - 0.5f) * w;
        t.y = (float(rand()%594) / 594.f - 0.5f) * h;
        t.z = (float(rand()%794) / 794.f - 0.5f) * d;
		
		*m_pool[i].t1 = t;
		*m_pool[i].t2 = t.normal();
		*m_pool[i].t3 = t;;
    }
	
	
	m_sculptor = new Sculptor;
	m_sculptor->beginAddVertices(2.f);
    
    VertexP * p = new VertexP[NUMVERTEX];
    for(i = 0; i < NUMVERTEX; i++) {
        p[i].key = i;
		p[i].index = &m_pool[i];
        
/// add vertex as P, N, Pref
        m_sculptor->insertVertex(&p[i]);
    }
	
	m_sculptor->endAddVertices();
	m_sculptor->setSelectRadius(2.5f);
}

GLWidget::~GLWidget()
{
	delete m_sculptor;
    delete[] m_pool;
}

void GLWidget::clientDraw()
{
	KdTreeDrawer * dr = getDrawer();
	BoundingBox bb;
	
	drawPoints(m_sculptor->allPoints());
	
	bb = m_sculptor->allPoints()->boundingBox();
	dr->boundingBox(bb);
	
	dr->setColor(0.f, 1.f, .3f);
	drawPoints(*m_sculptor->activePoints());
}

void GLWidget::drawPoints(WorldGrid<Array<int, VertexP>, VertexP > * tree)
{
    // KdTreeDrawer * dr = getDrawer();
	tree->begin();
	while(!tree->end()) {
        // dr->setColor(0.1 * (rand() & 7), 0.1 * (rand() & 7), 0.1 * (rand() & 7));
		drawPoints(tree->value());
		tree->next();
	}
}

void GLWidget::drawPoints(Array<int, VertexP> * d) 
{
	if(!d) return;
	KdTreeDrawer * dr = getDrawer();
	dr->beginPoint(2.f);
	
	d->begin();
	while(!d->end()) {
		Vector3F * p = d->value()->index->t1;
		dr->vertex(*p);
		d->next();
	}
	dr->end();
}

void GLWidget::drawPoints(const ActiveGroup & grp)
{
	Array<int, VertexP> * ps = grp.vertices;
	if(ps->size() < 1) return;
	drawPoints(ps);
}

void GLWidget::clientSelect(QMouseEvent */*event*/)
{
	setUpdatesEnabled(false);
	m_sculptor->clearCurrentStage();
	//m_sculptor->selectPoints(getIncidentRay());
	setUpdatesEnabled(true);
}

void GLWidget::clientDeselect(QMouseEvent */*event*/) 
{
	setUpdatesEnabled(false);
	m_sculptor->deselectPoints();
	setUpdatesEnabled(true);
}

void GLWidget::clientMouseInput(QMouseEvent */*event*/)
{
	setUpdatesEnabled(false);
	m_sculptor->selectPoints(getIncidentRay());
	// m_sculptor->pullPoints();
	const Vector3F dv = strokeVector(m_sculptor->activePoints()->minDepth());
	m_sculptor->smudgePoints(dv);
	setUpdatesEnabled(true);
}

void GLWidget::keyPressEvent(QKeyEvent *event)
{
	setUpdatesEnabled(false);
	switch (event->key()) {
		case Qt::Key_Z:
			if(event->modifiers() == Qt::ShiftModifier)
				m_sculptor->redo();
			else 
				m_sculptor->undo();
			
			break;
		default:
			break;
	}
	setUpdatesEnabled(true);
	Base3DView::keyPressEvent(event);
}
//:~