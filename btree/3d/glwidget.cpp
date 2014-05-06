#include <QtGui>
#include <QtOpenGL>
#include <Types.h>
#include "glwidget.h"
#include <cmath>
#include <KdTreeDrawer.h>
#define NUMVERTEX 1024
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
    int i;
	float w = 13.3f, h = 3.f, d = 13.f;
    m_pool = new V3[NUMVERTEX];
    for(i = 0; i < NUMVERTEX; i++) {
        V3 & t = m_pool[i];
		if(i > NUMVERTEX/2) {
			w = 12.f;
			h = 13.3f;
			d = 3.f;
		}
        t.data[0] = (float(rand()%694) / 694.f - 0.5f) * w;
        t.data[1] = (float(rand()%594) / 594.f - 0.5f) * h;
        t.data[2] = (float(rand()%794) / 794.f - 0.5f) * d;
    }
    
    m_tree = new C3Tree;
	m_tree->setGridSize(1.13f);
    
    VertexP p;
    for(i = 0; i < NUMVERTEX; i++) {
        p.key = i;
        p.index = &m_pool[i];
        m_tree->insert(p);
    }
    
    m_tree->display();
	
	i = 0;
	m_tree->firstGrid();
	while(!m_tree->gridEnd()) {
		i++;
		m_tree->nextGrid();
	}
	
	std::cout<<"grid count "<<i;
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
	Vector3F p;
	int i;
	dr->beginPoint(2.f);
	for(i = 0; i < NUMVERTEX; i++) {
        V3 & v = m_pool[i];
		p.set(v.data[0], v.data[1], v.data[2]);
		dr->vertex(p);
    }
	dr->end();
	
    m_tree->firstGrid();
	while(!m_tree->gridEnd()) {
		bb = m_tree->gridBoundingBox();
		dr->boundingBox(bb);
		m_tree->nextGrid();
	}
}