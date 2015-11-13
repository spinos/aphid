#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
    
    std::cout<<" test kdtree\n";
	const int n = 5000;
    m_source = new VectorArray<TestBox>();
	
	BoundingBox rootBox;
    int i;
    for(i=0; i<n; i++) {
        TestBox *a = new TestBox;
        float r = sqrt(float( rand() % 999 ) / 999.f);
        float th = float( rand() % 999 ) / 999.f * 1.5f;
        float x = -60.f + 200.f * r * cos(th);
        float y = -40.f + 100.f * r * sin(th) + 36.f * sin(x/12.f);
        float z = -40.f + 50.f * float( rand() % 999 ) / 999.f + 9.f * sin(y/23.f);
        a->setMin(-1 + x, -1 + y, -1 + z);
        a->setMax( 1 + x,  1 + y,  1 + z);
        
		m_source->add(a);
		rootBox.expandBy(a->calculateBBox());
    }
	
	m_engine.initGeometry(m_source, rootBox);
    m_tree = m_engine.tree();
	
	
	const double nearClip = 1.001f;
    const double farClip = 999.999f;
    const double hAperture = 1.f;
    const double vAperture = .75f;
    const double aov = 55.f;
	Matrix44F camspace;
	camspace.rotateX( .05f);
	camspace.rotateY(-.14f);
	camspace.setTranslation(10.f, 10.f, 170.f);
	m_frustum.set(nearClip, farClip, hAperture, vAperture, aov, camspace);
    
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
	getDrawer()->frustum(&m_frustum);
    // drawBoxes();
    drawTree();
    m_engine.render(m_frustum);
    drawScreen();
}

void GLWidget::drawBoxes() const
{
    getDrawer()->setColor(.065f, .165f, .065f);
    const int n = m_source->size();
    int i = 0;
    for(;i<n;i++) {
        getDrawer()->boundingBox(*m_source->get(i));
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
		drawALeaf(m_tree->leafPrimStart(nn->getPrimStart() ), nn->getNumPrims(), box);
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
	if(level >= m_maxDrawTreeLevel) return;
	
	if(level == m_maxDrawTreeLevel-1) {
		m_treeletColI++;
		getDrawer()->setGroupColorLight(m_treeletColI);
	}
	else getDrawer()->setColor(.1f, .15f, .12f);
	
	getDrawer()->boundingBox(lftBox);
    getDrawer()->boundingBox(rgtBox);
    
	drawANode(treelet, 0, lftBox, level);
	drawANode(treelet, 1, rgtBox, level);
	
	drawConnectedTreelet(treelet, 0, lftBox, level+1);
	drawConnectedTreelet(treelet, 1, rgtBox, level+1);
}

void GLWidget::drawALeaf(unsigned start, unsigned n, const BoundingBox & box)
{
	if(n<1) {
		// getDrawer()->setColor(0.f, 0.f, 0.f);
		// getDrawer()->boundingBox(box);
	}
	else {
		int i = 0;
		for(;i<n;i++) {
			getDrawer()->boundingBox(* m_tree->dataAt(start + i) );
		}
	}
}

void GLWidget::drawScreen()
{
    getDrawer()->setColor(0.f, .6f, 0.f);
    KdEngine<TestBox>::ScreenType * scrn = m_engine.screen();
    const unsigned n = scrn->m_views.size();
	// qDebug()<<" nv "<<n;
    unsigned i = 0;
    for(;i<n;i++) {
		getDrawer()->frustum(&scrn->m_views[i].view());
	}
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
			// qDebug()<<"down level "<<m_maxDrawTreeLevel;
		    break;
		case Qt::Key_L:
			m_maxDrawTreeLevel++;
		    // qDebug()<<"up level "<<m_maxDrawTreeLevel;
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

void GLWidget::resizeEvent(QResizeEvent * event)
{
    QSize renderAreaSize = size();
    qDebug()<<"render size "<<renderAreaSize.width()<<" "<<renderAreaSize.height();
    m_engine.initScreen(renderAreaSize.width(), renderAreaSize.height());
    Base3DView::resizeEvent(event);
}
