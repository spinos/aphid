#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include <KdBuilder.h>

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
    
    std::cout<<" test kdtree\n";
	const int n = 1100;
    m_boxes = new SahSplit<TestBox>(n);
	BoundingBox rootBox;
    int i;
    for(i=0; i<n; i++) {
        TestBox *a = new TestBox;
        float r = sqrt(float( rand() % 999 ) / 999.f);
        float th = float( rand() % 999 ) / 999.f * 1.5f;
        float x = -60.f + 120.f * r * cos(th);
        float y = -40.f + 80.f * r * sin(th);
        float z = -40.f + 22.f * float( rand() % 999 ) / 999.f + 5.f * sin(y/23.f);
        a->setMin(-1 + x, -1 + y, -1 + z);
        a->setMax( 1 + x,  1 + y,  1 + z);
        m_boxes->set(i, a);
		rootBox.expandBy(a->calculateBBox());
    }
    m_boxes->setBBox(rootBox);
    
    m_tree = new KdNTree<TestBox, KdNode4 >(n);
    m_tree->setBBox(rootBox);
    
    std::cout<<" max n nodes "<<m_tree->maxNumNodes();
	
	KdNBuilder<4, 4, TestBox, KdNode4 > bud;
	bud.build(m_boxes, m_tree->nodes());
    std::cout<<" tree bound "<<m_tree->getBBox();
    
    m_tree->nodes()[0].verbose();
    std::cout<<"\n size of node "<<sizeof(KdNode4);
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
	connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
    drawBoxes();
    drawTree();
}

void GLWidget::drawBoxes() const
{
    getDrawer()->setColor(.65f, .65f, .65f);
    const int n = m_boxes->numPrims();
    int i = 0;
    for(;i<n;i++) {
        getDrawer()->boundingBox(*m_boxes->get(i));
    }
}

void GLWidget::drawTree()
{
    drawNode(m_tree->nodes(), 0, m_tree->getBBox() );
}

void GLWidget::drawNode(KdNode4 * nodes, int idx, const BoundingBox & box)
{
    getDrawer()->setColor(.15f, .25f, .35f);
    getDrawer()->boundingBox(box);
    KdNode4 * n = &nodes[idx];
    int i = 0;
    for(;i<KdNode4::NumNodes;i++) {
        KdTreeNode * nn = n->node(i);
        if(nn->isLeaf()) continue;
        drawSplitPlane(nn, box);
    }
}

void GLWidget::drawSplitPlane(KdTreeNode * node, const BoundingBox & box)
{
    Vector3F corner0(box.getMin(0), box.getMin(1), box.getMin(2));
	Vector3F corner1(box.getMax(0), box.getMax(1), box.getMax(2));
	const int axis = node->getAxis();
    const float pos = node->getSplitPos();
	corner0.setComp(pos, axis);
	corner1.setComp(pos, axis);
    glBegin(GL_LINE_LOOP);
	if(axis == 0) {
		glVertex3f(corner0.x, corner0.y, corner0.z);
		glVertex3f(corner0.x, corner1.y, corner0.z);
		glVertex3f(corner0.x, corner1.y, corner1.z);
		glVertex3f(corner0.x, corner0.y, corner1.z);
	}
	else if(axis == 1) {
		glVertex3f(corner0.x, corner0.y, corner0.z);
		glVertex3f(corner1.x, corner0.y, corner0.z);
		glVertex3f(corner1.x, corner0.y, corner1.z);
		glVertex3f(corner0.x, corner0.y, corner1.z);
	}
	else {
		glVertex3f(corner0.x, corner0.y, corner0.z);
		glVertex3f(corner1.x, corner0.y, corner0.z);
		glVertex3f(corner1.x, corner1.y, corner0.z);
		glVertex3f(corner0.x, corner1.y, corner0.z);
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

void GLWidget::simulate()
{
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	Base3DView::keyPressEvent(e);
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base3DView::keyReleaseEvent(event);
}

