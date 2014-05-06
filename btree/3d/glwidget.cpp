#include <QtGui>
#include <QtOpenGL>
#include <Types.h>
#include "glwidget.h"
#include <cmath>
#define NUMVERTEX 128
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
    int i;
    m_pool = new V3[NUMVERTEX];
    for(i = 0; i < NUMVERTEX; i++) {
        V3 & t = m_pool[i];
        t.data[0] = (float(rand()%694) / 694.f - 0.5f) * 32.4f;
        t.data[1] = (float(rand()%594) / 594.f - 0.5f) * 32.4f;
        t.data[2] = (float(rand()%794) / 794.f - 0.5f) * 32.4f;
    }
    
    m_tree = new C3Tree;
    
    VertexP p;
    for(i = 0; i < NUMVERTEX; i++) {
        p.key = i;
        p.index = &m_pool[i];
        m_tree->insert(p);
    }
    
    m_tree->display();
}

GLWidget::~GLWidget()
{
    delete m_tree;
    delete[] m_pool;
}

