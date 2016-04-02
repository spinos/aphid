#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "wldWidget.h"
#include <GeoDrawer.h>
#include "NTreeDrawer.h"
#include <NTreeIO.h>
#include <HWorldGrid.h>
#include <HInnerGrid.h>
#include "GridDrawer.h"

using namespace aphid;

WldWidget::WldWidget(const std::string & filename, QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
    m_intersectCtx.m_success = 0;
	
	m_source = NULL;
	m_tree = NULL;
	
	if(filename.size() > 1) readTree(filename);
}
//! [0]

//! [1]
WldWidget::~WldWidget()
{
	delete m_source;
	delete m_tree;
}

void WldWidget::clientInit()
{
	connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void WldWidget::clientDraw()
{
	drawBoxes();
	drawTree();
	drawIntersect();
}

void WldWidget::drawBoxes() const
{
	if(!m_source) return;
    getDrawer()->setColor(.065f, .165f, .065f);
	NTreeDrawer dr;
	dr.drawSource<cvx::Cube>(m_tree);
}

void WldWidget::drawTree()
{
	if(!m_tree) return; 
	
	getDrawer()->setColor(.15f, .25f, .35f);
	getDrawer()->boundingBox(tree()->getBBox() );
	
	NTreeDrawer dr;
	dr.drawTree<cvx::Cube>(m_tree);
}

KdNTree<cvx::Cube, KdNode4 > * WldWidget::tree()
{ return m_tree; }

void WldWidget::clientSelect(QMouseEvent *event)
{
	setUpdatesEnabled(false);
	testIntersect(getIncidentRay());
	setUpdatesEnabled(true);
}

void WldWidget::clientDeselect(QMouseEvent *event)
{}

void WldWidget::clientMouseInput(QMouseEvent *event)
{
	setUpdatesEnabled(false);
	testIntersect(getIncidentRay());
	setUpdatesEnabled(true);
}

void WldWidget::keyPressEvent(QKeyEvent *event)
{
	switch (event->key()) {
		case Qt::Key_F:
			camera()->frameAll(getFrameBox() );
		    break;
		default:
			break;
	}
	
	Base3DView::keyPressEvent(event);
}

void WldWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base3DView::keyReleaseEvent(event);
}

void WldWidget::resizeEvent(QResizeEvent * event)
{
    QSize renderAreaSize = size();
    // qDebug()<<"render size "<<renderAreaSize.width()<<" "<<renderAreaSize.height();
    Base3DView::resizeEvent(event);
}

bool WldWidget::readTree(const std::string & filename)
{
	bool stat = false;
	NTreeIO hio;
	if(!hio.begin(filename) ) return false;
	
	std::string gridName;
	stat = hio.findGrid(gridName);
	if(stat) {
		std::cout<<"\n grid "<<gridName;
		m_source = new sdb::VectorArray<cvx::Cube>();
		hio.loadGridCoord<sdb::HWorldGrid<sdb::HInnerGrid<hdata::TFloat, 4, 1024 >, cvx::Sphere > >(m_source, gridName);
	} else
		std::cout<<"\n found no grid ";
	
	// cvx::ShapeType vt = hio.gridValueType(gridName);
    
	std::string treeName;
	stat = hio.findTree(treeName, gridName);
	if(stat) {
		std::cout<<"\n tree "<<treeName;
		HNTree<cvx::Cube, KdNode4 > * htree = new HNTree<cvx::Cube, KdNode4 >(treeName);
		htree->load();
		htree->close();
		htree->setSource(m_source);
		m_tree = htree;
	} else
		std::cout<<"\n found no tree ";
	
	hio.end();
	return true;
}

void WldWidget::testIntersect(const Ray * incident)
{
	m_intersectCtx.reset(*incident);
	if(!m_tree) return;
	std::stringstream sst; sst<<incident->m_dir;
	qDebug()<<"interset begin "<<sst.str().c_str();
	m_tree->intersect(&m_intersectCtx);
	qDebug()<<"interset end";
}

void WldWidget::drawIntersect()
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
	
	if(m_intersectCtx.m_success) drawActiveSource(m_intersectCtx.m_leafIdx);
}

void WldWidget::drawActiveSource(const unsigned & iLeaf)
{
	if(!m_tree) return;
	if(!m_source) return;
	
	glColor3f(0,.6,.4);
	int start, len;
	m_tree->leafPrimStartLength(start, len, iLeaf);
	int i=0;
	for(;i<len;++i) {
		const cvx::Cube * c = m_source->get( m_tree->primIndirectionAt(start + i) );
		BoundingBox b = c->calculateBBox();
		b.expand(-0.03f);
		getDrawer()->boundingBox(b );
	}
}

BoundingBox WldWidget::getFrameBox()
{
	BoundingBox b;
	if(m_tree) b = m_tree->getBBox();
	return b;
}
//:~