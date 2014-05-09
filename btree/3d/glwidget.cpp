#include <QtGui>
#include <QtOpenGL>
#include <Types.h>
#include "glwidget.h"
#include <cmath>
#include <KdTreeDrawer.h>
#include <Sequence.h>

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
    
    m_tree = new C3Tree;
	m_tree->setGridSize(1.43f);
    
    VertexP p;
    for(i = 0; i < NUMVERTEX; i++) {
        p.key = i;
        p.index = &m_pool[i];
        m_tree->insert(p);
    }
    
    //m_tree->display();
	
	i = 0;
	m_tree->firstGrid();
	while(!m_tree->gridEnd()) {
		i++;
		m_tree->nextGrid();
	}
	
	std::cout<<"grid count "<<i;
	m_tree->calculateBBox();
	
	m_march.initialize(m_tree->boundingBox(), m_tree->gridSize());
	m_rayBegin.set(-20.f, 5.f, 20.f);
	m_rayEnd.set(12.f, -2.f, -11.f);
}

GLWidget::~GLWidget()
{
    delete m_tree;
    delete[] m_pool;
}

void GLWidget::clientDraw()
{
	KdTreeDrawer * dr = getDrawer();
	BoundingBox bb;

	m_tree->firstGrid();
	while(!m_tree->gridEnd()) {
		drawPoints(m_tree->verticesInGrid());
		m_tree->nextGrid();
	}
	
    m_tree->firstGrid();
	while(!m_tree->gridEnd()) {
		bb = m_tree->gridBoundingBox();
		//dr->boundingBox(bb);
		m_tree->nextGrid();
	}
	
	bb = m_tree->boundingBox();
	dr->boundingBox(bb);
	
	std::vector<Vector3F> linevs;
	linevs.push_back(m_rayBegin);
	linevs.push_back(m_rayEnd);
	
	dr->setColor(0.f, 1.f, 0.f);
	dr->lines(linevs);
	
	Sequence<Coord3> added;
	List<VertexP> intube;
	Ray inc(m_rayBegin, m_rayEnd);
	if(!m_march.begin(inc)) return;
	while(!m_march.end()) {
		const std::deque<Vector3F> coords = m_march.touched(1.82f);
		std::deque<Vector3F>::const_iterator it = coords.begin();
		for(; it != coords.end(); ++it) {
			const Coord3 c = m_tree->gridCoord((const float *)&(*it));
			if(added.find(c)) continue;
			added.insert(c);
			List<VertexP> * pl = m_tree->find((float *)&(*it));
			intersect(pl, inc, 1.82f, intube);
			//if(pl) {
				//if(intersect(pl, inc, 1.82f, intube)) { 
				//	dr->boundingBox(m_march.computeBBox(*it));
				//}
			//}
		}
		m_march.step();
	}
	drawPoints(&intube);
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

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	if(e->key() == Qt::Key_N) {
		m_rayBegin.set(-20.f + (float(rand()%694) / 694.f - 0.5f) * 5.f, 5.f + (float(rand()%694) / 694.f - 0.5f) * 15.f, 20.f + (float(rand()%694) / 694.f - 0.5f) * 1.f);
		m_rayEnd.set(12.f + (float(rand()%694) / 694.f - 0.5f) * 15.f, -2.f + (float(rand()%694) / 694.f - 0.5f) * 25.f, -11.f + (float(rand()%694) / 694.f - 0.5f) * 15.f);
	}
	Base3DView::keyPressEvent(e);
}

bool GLWidget::intersect(const List<VertexP> * d, const Ray & ray, const float & threshold, List<VertexP> & dst)
{
	if(!d) return false;
	const int num = d->size();
	const int ndst = dst.size();
	Vector3F p, pop;
	for(int i = 0; i < num; i++) {
		V3 * v = d->value(i).index;
		p.set(v->data[0], v->data[1], v->data[2]);
		float tt = ray.m_origin.dot(ray.m_dir) - p.dot(ray.m_dir);
		pop = ray.m_origin - ray.m_dir * tt;
		if(p.distanceTo(pop) < threshold)
			dst.insert(d->value(i));
	}
	return dst.size() > ndst;
}

