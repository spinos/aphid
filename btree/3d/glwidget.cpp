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
    m_pos = new Vector3F[NUMVERTEX];
	m_nor = new Vector3F[NUMVERTEX];
	m_ref = new Vector3F[NUMVERTEX];
    for(i = 0; i < NUMVERTEX; i++) {
        Vector3F & t = m_pos[i];
		if(i > NUMVERTEX/2) {
			w = 5.f;
			h = 16.3f;
			d = 3.f;
		}
        t.x = (float(rand()%694) / 694.f - 0.5f) * w;
        t.y = (float(rand()%594) / 594.f - 0.5f) * h;
        t.z = (float(rand()%794) / 794.f - 0.5f) * d;
		
		m_nor[i] = m_pos[i]; m_nor[i].normalize();
		m_ref[i] = m_pos[i];
    }
	
	m_pool = new PNPref[NUMVERTEX];
	
	m_sculptor = new Sculptor;
	m_sculptor->beginAddVertices(2.f);
    
    VertexP p;
    for(i = 0; i < NUMVERTEX; i++) {
        p.key = i;
		p.index = &m_pool[i];
        (p.index)->t1 = &m_pos[i];
		(p.index)->t2 = &m_nor[i];
		(p.index)->t3 = &m_ref[i];
        m_sculptor->addVertex(p);
    }
	
	m_sculptor->endAddVertices();
	m_sculptor->setSelectRadius(2.f);
}

GLWidget::~GLWidget()
{
	delete m_sculptor;
    delete[] m_pool;
	delete[] m_pos;
	delete[] m_nor;
	delete[] m_ref;
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
    qDebug()<<"";
}

void GLWidget::drawPoints(C3Tree * tree)
{
    KdTreeDrawer * dr = getDrawer();
	tree->begin();
	while(!tree->end()) {
        // dr->setColor(0.1 * (rand() & 7), 0.1 * (rand() & 7), 0.1 * (rand() & 7));
		drawPoints(tree->verticesInGrid());
		tree->next();
	}
}

void GLWidget::drawPoints(List<VertexP> * d) 
{
	if(!d) return;
	KdTreeDrawer * dr = getDrawer();
	dr->beginPoint(2.f);
	
	d->begin();
	while(!d->end()) {
		Vector3F * p = d->value().index->t1;
		dr->vertex(*p);
		d->next();
	}
	dr->end();
}

void GLWidget::drawPoints(const Sculptor::ActiveGroup & grp)
{
	Ordered<int, VertexP> * ps = grp.vertices;
	if(ps->size() < 1) return;
	const int maxNumBlk = grp.numActiveBlocks();
	int blk = 0;
	ps->begin();
	while(!ps->end()) {
		List<VertexP> * vs = ps->value();
		drawPoints(vs);
		blk++;
		if(blk == maxNumBlk) return;
		ps->next();
	}
}

void GLWidget::clientSelect(QMouseEvent */*event*/)
{
	setUpdatesEnabled(false);
	m_sculptor->selectPoints(getIncidentRay());
	// m_sculptor->pullPoints();
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
	const Vector3F dv = strokeVector(m_sculptor->activePoints()->meanDepth());
	m_sculptor->smudgePoints(dv);
	setUpdatesEnabled(true);
}
