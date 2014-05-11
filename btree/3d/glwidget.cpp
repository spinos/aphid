#include <QtGui>
#include <QtOpenGL>
#include <Types.h>
#include "glwidget.h"
#include <cmath>
#include <KdTreeDrawer.h>
#include <Sequence.h>
#include <Ordered.h>
#define NUMVERTEX 12000

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
	m_sculptor->beginAddVertices(1.51f);
    
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
	m_sculptor->setSelectRadius(1.97f);
	
	std::vector<X> xs;
	X ax; ax.a = 10; ax.b = 11;
	xs.push_back(ax);
	std::cout<<"xs[0]:"<<xs[0].a<<","<<xs[0].b<<"\n";
	X * bx = &xs[0];
	bx->a = 9; bx->b = 8;
	std::cout<<"xs[0]:"<<xs[0].a<<","<<xs[0].b<<"\n";
	
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

	for(int i = 0; i < num; i++) {
		Vector3F * p = d->value(i).index->t1;
		dr->vertex(*p);
	}
	dr->end();
}

void GLWidget::drawPoints(const Sculptor::ActiveGroup & grp)
{
	Ordered<int, VertexP> * ps = grp.vertices;
	if(ps->size() < 1) return;
	ps->begin();
	int nblk = 0;
	while(!ps->end()) {
		const List<VertexP> * vs = ps->value();
		if(nblk >= grp.numActiveBlocks) return;
		drawPoints(vs);
		
		nblk++;
		ps->next();
	}
}

void GLWidget::clientSelect(QMouseEvent */*event*/)
{
	setUpdatesEnabled(false);
	m_sculptor->selectPoints(getIncidentRay());
	m_sculptor->pullPoints();
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
	m_sculptor->pullPoints();
	setUpdatesEnabled(true);
}
