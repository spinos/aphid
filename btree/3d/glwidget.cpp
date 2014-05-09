#include <QtGui>
#include <QtOpenGL>
#include <Types.h>
#include "glwidget.h"
#include <cmath>
#include <KdTreeDrawer.h>
#include <Sequence.h>
#include <Ordered.h>
#define NUMVERTEX 12000
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
    int i;
	float w = 29.3f, h = 3.f, d = 13.f;
    m_pool = new V3[NUMVERTEX];
    for(i = 0; i < NUMVERTEX; i++) {
        V3 & t = m_pool[i];
		if(i > NUMVERTEX/2) {
			w = 5.f;
			h = 16.3f;
			d = 3.f;
		}
        t.data[0] = (float(rand()%694) / 694.f - 0.5f) * w;
        t.data[1] = (float(rand()%594) / 594.f - 0.5f) * h;
        t.data[2] = (float(rand()%794) / 794.f - 0.5f) * d;
    }
	
	m_sculptor = new Sculptor;
	m_sculptor->beginAddVertices(1.51f);
    
    VertexP p;
    for(i = 0; i < NUMVERTEX; i++) {
        p.key = i;
        p.index = &m_pool[i];
        m_sculptor->addVertex(p);
    }
	
	m_sculptor->endAddVertices();
	
	m_sculptor->setSelectRadius(1.97f);
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

void GLWidget::drawPoints(C3Tree * tree)
{
	tree->begin();
	while(!tree->end()) {
		drawPoints(tree->verticesInGrid());
		tree->next();
	}
}

void GLWidget::drawPoints(const List<VertexP> * d) 
{
	if(!d) return;
	KdTreeDrawer * dr = getDrawer();
	dr->beginPoint(2.f);
	const int num = d->size();
	VertexP v;
	Vector3F p;
	for(int i = 0; i < num; i++) {
		V3 * v = d->value(i).index;
		p.set(v->data[0], v->data[1], v->data[2]);
		dr->vertex(p);
	}
	dr->end();
}

void GLWidget::drawPoints(const Sculptor::ActiveGroup & grp)
{
	Ordered<int, VertexP> * ps = grp.vertices;
	if(ps->size() < 1) return;
	const float mxr = grp.threshold * 2.f;
	ps->begin();
	while(!ps->end()) {
		const List<VertexP> * vs = ps->value();
		if((ps->key() - 1) * grp.gridSize - grp.depthMin > mxr) return;
		drawPoints(vs);
		ps->next();
	}
}

void GLWidget::clientSelect(QMouseEvent */*event*/)
{
	m_sculptor->selectPoints(getIncidentRay());
}

void GLWidget::clientDeselect(QMouseEvent */*event*/) 
{
	m_sculptor->deselectPoints();
}

void GLWidget::clientMouseInput(QMouseEvent */*event*/)
{
	m_sculptor->selectPoints(getIncidentRay());
}
