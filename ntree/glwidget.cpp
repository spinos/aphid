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
	const int n = 4000;
    m_boxes = new SahSplit<TestBox>(n);
	BoundingBox rootBox;
    int i;
    for(i=0; i<n; i++) {
        TestBox *a = new TestBox;
        float r = sqrt(float( rand() % 999 ) / 999.f);
        float th = float( rand() % 999 ) / 999.f * 1.5f;
        float x = -60.f + 200.f * r * cos(th);
        float y = -40.f + 100.f * r * sin(th) + 5.f * sin(x/23.f);
        float z = -40.f + 50.f * float( rand() % 999 ) / 999.f + 5.f * sin(y/23.f);
        a->setMin(-1 + x, -1 + y, -1 + z);
        a->setMax( 1 + x,  1 + y,  1 + z);
        m_boxes->set(i, a);
		rootBox.expandBy(a->calculateBBox());
    }
    m_boxes->setBBox(rootBox);
    
    m_tree = new KdNTree<TestBox, KdNode4 >(n);
    m_tree->setBBox(rootBox);
    
    std::cout<<" max n nodes "<<m_tree->maxNumNodes();
	
	KdNBuilder<4, TestBox, KdNode4 > bud;
	bud.SetNumPrimsInLeaf(9);
	bud.build(m_boxes, m_tree->nodes());
    m_maxDrawTreeLevel = 1;
    // std::cout<<"\n size of node "<<sizeof(KdNode4);
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
    getDrawer()->setColor(.065f, .165f, .065f);
    const int n = m_boxes->numPrims();
    int i = 0;
    for(;i<n;i++) {
        getDrawer()->boundingBox(*m_boxes->get(i));
    }
}

void GLWidget::drawTree()
{
	m_treeletColI = 0;
	getDrawer()->setColor(.15f, .25f, .35f);
	getDrawer()->boundingBox(m_tree->getBBox() );
    drawANode(&m_tree->nodes()[0], 0, m_tree->getBBox(), 0, true );
}

void GLWidget::drawANode(KdNode4 * treelet, int idx, const BoundingBox & box, int level, bool isRoot)
{
	if(level == m_maxDrawTreeLevel-1) getDrawer()->setGroupColorLight(m_treeletColI);
	else getDrawer()->setColor(.1f, .15f, .12f);
	
	KdTreeNode * nn = treelet->node(idx);
	if(nn->isLeaf()) {
		// getDrawer()->boundingBox(box);
		return;
	}
	
	BoundingBox flat(box);
	const int axis = nn->getAxis();
	const float pos = nn->getSplitPos();
	flat.setMin(pos, axis);
	flat.setMax(pos, axis);
	
	if(idx < Treelet4::LastLevelOffset() ) getDrawer()->boundingBox(flat);
	
	BoundingBox lft, rgt;
	box.split(axis, pos, lft, rgt);
	
	int offset = nn->getOffset();
	if(offset > KdNode4::TreeletOffsetMask ) {
		offset &= ~KdNode4::TreeletOffsetMask;
		if(isRoot) drawATreelet(treelet + offset, lft, rgt, level);
	}
	else {
		drawANode(treelet, idx + offset, lft, level);
		drawANode(treelet, idx + offset + 1, rgt, level);
	}
}

void GLWidget::drawConnectedTreelet(KdNode4 * treelet, int idx, const BoundingBox & box, int level)
{
	KdTreeNode * nn = treelet->node(idx);
	if(nn->isLeaf()) return;
	
	const int axis = nn->getAxis();
	const float pos = nn->getSplitPos();
	BoundingBox lft, rgt;
	box.split(axis, pos, lft, rgt);
	
	int offset = nn->getOffset();
	if(offset > KdNode4::TreeletOffsetMask ) {
		offset &= ~KdNode4::TreeletOffsetMask;
		drawATreelet(treelet + offset, lft, rgt, level);
	}
	else {
		drawConnectedTreelet(treelet, idx + offset, lft, level);
		drawConnectedTreelet(treelet, idx + offset + 1, rgt, level);
	}
}

void GLWidget::drawATreelet(KdNode4 * treelet, const BoundingBox & lftBox, const BoundingBox & rgtBox, int level)
{	
	m_treeletColI++;
	if(level >= m_maxDrawTreeLevel) return;
	
	if(level == m_maxDrawTreeLevel-1) getDrawer()->setGroupColorLight(m_treeletColI);
	else getDrawer()->setColor(.1f, .15f, .12f);
	
	getDrawer()->boundingBox(lftBox);
    getDrawer()->boundingBox(rgtBox);
    
	drawANode(treelet, 0, lftBox, level);
	drawANode(treelet, 1, rgtBox, level);
	
	drawConnectedTreelet(treelet, 0, lftBox, level+1);
	drawConnectedTreelet(treelet, 1, rgtBox, level+1);
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

void GLWidget::keyPressEvent(QKeyEvent *event)
{
	switch (event->key()) {
		case Qt::Key_K:
			m_maxDrawTreeLevel--;
			qDebug()<<"down level "<<m_maxDrawTreeLevel;
		    break;
		case Qt::Key_L:
			m_maxDrawTreeLevel++;
		    qDebug()<<"up level "<<m_maxDrawTreeLevel;
			break;
		default:
			break;
	}
	
	Base3DView::keyPressEvent(event);
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base3DView::keyReleaseEvent(event);
}

