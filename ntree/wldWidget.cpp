#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "wldWidget.h"
#include <GeoDrawer.h>
#include "NTreeDrawer.h"
#include "GridDrawer.h"

using namespace aphid;

WldWidget::WldWidget(const std::string & filename, QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
    m_intersectCtx.m_success = 0;
	
	m_grid = NULL;
	m_source = NULL;
	m_tree = NULL;
	m_voxel = NULL;
	
	if(filename.size() > 1) readTree(filename);
}
//! [0]

//! [1]
WldWidget::~WldWidget()
{
	if(m_grid) delete m_grid;
	if(m_source) delete m_source;
	if(m_tree) delete m_tree;
}

void WldWidget::clientInit()
{// connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update())); 
}

void WldWidget::clientDraw()
{
	drawBoxes();
	drawTree();
	drawIntersect();
	drawVoxel();
}

void WldWidget::drawBoxes() const
{
	if(!m_source) return;
    getDrawer()->setColor(.0f, .065f, .165f);
	NTreeDrawer dr;
	dr.drawSource<cvx::Box>(m_tree);
}

void WldWidget::drawTree()
{
	if(!m_tree) return; 
	
	//getDrawer()->setColor(.15f, .25f, .35f);
	//getDrawer()->boundingBox(tree()->getBBox() );
	
	NTreeDrawer dr;
	dr.drawTree<cvx::Box>(m_tree);
}

KdNTree<cvx::Box, KdNode4 > * WldWidget::tree()
{ return m_tree; }

void WldWidget::clientSelect(QMouseEvent *event)
{
	setUpdatesEnabled(false);
	testIntersect(getIncidentRay());
	setUpdatesEnabled(true);
	update();
}

void WldWidget::clientDeselect(QMouseEvent *event)
{}

void WldWidget::clientMouseInput(QMouseEvent *event)
{
	setUpdatesEnabled(false);
	testIntersect(getIncidentRay());
	setUpdatesEnabled(true);
	update();
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
	
	if(!m_hio.begin(filename) ) return false;
	
	stat = m_hio.objectExists("/grid/.tree");
	
	if(stat) {
		m_grid = new WorldGridT("/grid");
		m_grid->load();
		m_source = new sdb::VectorArray<cvx::Box>;
		if(m_hio.loadBox(m_source, m_grid) < 1) std::cout<<"\n error no box";
		m_tree = new HNTree<cvx::Box, KdNode4 >("/grid/.tree");
		m_tree->load();
		m_tree->close();
		m_tree->setSource(m_source);
		
	} else {
		std::cout<<"\n  found no grid ";
	}
	
	return true;
}

void WldWidget::testIntersect(const Ray * incident)
{
	m_intersectCtx.reset(*incident);
	if(!m_tree) return;
	std::stringstream sst; sst<<incident->m_dir;
	qDebug()<<"interset begin "<<sst.str().c_str();
	KdEngine eng;
	eng.intersect<cvx::Box, KdNode4>(m_tree, &m_intersectCtx);
	qDebug()<<"interset end";
	if(m_intersectCtx.m_success) {
		qDebug()<<" hit component "<<m_intersectCtx.m_componentIdx;
		m_voxel = m_grid->cell(m_intersectCtx.m_componentIdx)->loadTree();
	}
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
		const cvx::Box * c = m_source->get( m_tree->primIndirectionAt(start + i) );
		BoundingBox b = c->calculateBBox();
		b.expand(-0.03f);
		getDrawer()->boundingBox(b );
	}
}

void WldWidget::drawVoxel()
{
	if(!m_voxel) return;
	glColor3f(.0f,.65f,.45f);
	NTreeDrawer dr;
	dr.drawTree<Voxel>(m_voxel);
	getDrawer()->setColor(0.f, .15f, .35f);
	dr.drawTightBox<Voxel>(m_voxel);
}

BoundingBox WldWidget::getFrameBox()
{
	BoundingBox b;
	if(m_tree) b = m_tree->getBBox();
	return b;
}
//:~