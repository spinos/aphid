#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <GeoDrawer.h>
#include <NTreeDrawer.h>
#include <NTreeIO.h>

GLWidget::GLWidget(const std::string & filename, QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
    m_intersectCtx.m_success = 0;
	
	m_maxDrawTreeLevel = 1;
	
	m_source = NULL;
	m_tree = NULL;
	
	std::cout<<"\n sizeof 4node "<<sizeof(KdNode4);
	if(filename.size() > 1) readTree(filename);
	else testTree();
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
	delete m_source;
	delete m_tree;
}

void GLWidget::testTree()
{
	std::cout<<"\n test kdtree";
	const int n = 16512;
    m_source = new sdb::VectorArray<cvx::Cube>();
	m_tree = new KdNTree<cvx::Cube, KdNode4 >();
	
	BoundingBox rootBox;
    int i;
    for(i=0; i<n; i++) {
        cvx::Cube a;
        float r = sqrt(float( rand() % 999 ) / 999.f);
        float th = float( rand() % 999 ) / 999.f * 1.5f;
        float x = -60.f + 100.f * r * cos(th*1.1f);
        float y = 0.f + 70.f * r * sin(th/.93f) + 39.f * sin(x/13.f);
        float z = 0.f + 50.f * float( rand() % 999 ) / 999.f + 23.f * sin(y/23.f);
        a.set(Vector3F(x, y, z), .2f);
        
		m_source->insert(a);
		rootBox.expandBy(a.calculateBBox());
    }
	
    TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 8;
    m_engine.buildTree(m_tree, m_source, rootBox, &bf);
	// m_engine.printTree(m_tree);
}

void GLWidget::clientInit()
{
	connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
	// getDrawer()->frustum(&m_frustum);
	drawBoxes();
    drawTree();
	drawIntersect();
}

void GLWidget::drawBoxes() const
{
	if(!m_source) return;
    getDrawer()->setColor(.065f, .165f, .065f);
    const int n = m_source->size();
    int i = 0;
    for(;i<n;i++) {
        getDrawer()->boundingBox(m_source->get(i)->calculateBBox() );
    }
}

void GLWidget::drawTree()
{
	if(!m_tree) return; 
	m_treeletColI = 0;
	getDrawer()->setColor(.15f, .25f, .35f);
	getDrawer()->boundingBox(tree()->getBBox() );
	
	NTreeDrawer dr;
	dr.drawTree<cvx::Cube>(m_tree);
}

void GLWidget::drawANode(KdNode4 * treelet, int idx, const BoundingBox & box, int level, bool isRoot)
{
	if(level == m_maxDrawTreeLevel-1) getDrawer()->setGroupColorLight(m_treeletColI);
	else getDrawer()->setColor(.1f, .15f, .12f);
	
	KdTreeNode * nn = treelet->node(idx);
	if(nn->isLeaf()) {
		std::cout<<"\n i "<<idx
			<<" leaf start "<<nn->getPrimStart()<<" n "<<nn->getNumPrims();
		drawALeaf(tree()->leafPrimStart(nn->getPrimStart() ), nn->getNumPrims(), box);
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
		if(offset> 0) {
		drawANode(treelet, idx + offset, lft, level);
		drawANode(treelet, idx + offset + 1, rgt, level);
		}
	}
}

void GLWidget::drawConnectedTreelet(KdNode4 * treelet, int idx, const BoundingBox & box, int level)
{
	KdTreeNode * nn = treelet->node(idx);
	if(nn->isLeaf() ) return;
	
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
		if(offset> 0) {
		drawConnectedTreelet(treelet, idx + offset, lft, level);
		drawConnectedTreelet(treelet, idx + offset + 1, rgt, level);
		}
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
			getDrawer()->boundingBox(tree()->dataAt(start + i)->calculateBBox() );
		}
	}
}

KdNTree<cvx::Cube, KdNode4 > * GLWidget::tree()
{ return m_tree; }

void GLWidget::clientSelect(QMouseEvent *event)
{
	setUpdatesEnabled(false);
	testIntersect(getIncidentRay());
	setUpdatesEnabled(true);
}

void GLWidget::clientDeselect(QMouseEvent *event)
{}

void GLWidget::clientMouseInput(QMouseEvent *event)
{
	setUpdatesEnabled(false);
	testIntersect(getIncidentRay());
	setUpdatesEnabled(true);
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
    // qDebug()<<"render size "<<renderAreaSize.width()<<" "<<renderAreaSize.height();
    Base3DView::resizeEvent(event);
}

bool GLWidget::readTree(const std::string & filename)
{
	bool stat = false;
	NTreeIO hio;
	if(!hio.begin(filename) ) return false;
	
	std::string gridName;
	stat = hio.findGrid(gridName);
	if(stat) {
		std::cout<<"\n grid "<<gridName;
		m_source = new sdb::VectorArray<cvx::Cube>();
		hio.loadSphereGridCoord(m_source, gridName);
	}
	
	cvx::ShapeType vt = hio.gridValueType(gridName);
    
	std::string treeName;
	stat = hio.findTree(treeName, gridName);
	if(stat) {
		m_tree= hio.loadCube4Tree(treeName);
		m_tree->setSource(m_source);
	}
	
	hio.end();
	return true;
}

void GLWidget::testIntersect(const Ray * incident)
{
	m_intersectCtx.reset(*incident);
	if(!m_tree) return;
	std::stringstream sst; sst<<incident->m_dir;
	qDebug()<<"interset begin "<<sst.str().c_str();
	m_tree->intersect(&m_intersectCtx);
	qDebug()<<"interset end";
}

void GLWidget::drawIntersect()
{
	Vector3F dst;
	if(m_intersectCtx.m_success) {
		glColor3f(0,1,0);
		dst = m_intersectCtx.m_ray.travel(m_intersectCtx.m_tmax);
	}
	else {
		glColor3f(1,0,0);
		dst = m_intersectCtx.m_ray.destination();
	}
	
	glBegin(GL_LINES);
		glVertex3fv((const GLfloat * )&m_intersectCtx.m_ray.m_origin);
		glVertex3fv((const GLfloat * )&dst);
	glEnd();
	
	BoundingBox b = m_intersectCtx.getBBox();
	b.expand(0.03f);
	getDrawer()->boundingBox(b );
	
	if(m_intersectCtx.m_success) drawActiveSource(m_intersectCtx.m_componentIdx);
}

void GLWidget::drawActiveSource(const unsigned & iLeaf)
{
	if(!m_tree) return;
	if(!m_source) return;
	
	int start, len;
	m_tree->leafPrimStartLength(start, len, iLeaf);
	int i=0;
	for(;i<len;++i) {
		const cvx::Cube * c = m_source->get( m_tree->primIndirectionAt(start + i) );
		getDrawer()->boundingBox(c->calculateBBox() );
	}
}
//:~