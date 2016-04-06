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
	
	m_voxelTree = NULL;
	
	if(filename.size() > 1) m_hio.openWorld(filename);
}

WldWidget::~WldWidget()
{}

void WldWidget::clientInit()
{}// connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));

void WldWidget::clientDraw()
{
	drawBoxes();
	drawTree();
	drawIntersect();
	drawVoxel();
}

void WldWidget::drawBoxes() const
{
	if(!m_hio.grid()) return;
    getDrawer()->setColor(.0f, .065f, .165f);
	NTreeDrawer dr;
	dr.drawSource<cvx::Box>(m_hio.grid()->tree() );
}

void WldWidget::drawTree()
{
	if(!m_hio.grid()) return; 
	
	NTreeDrawer dr;
	dr.drawTree<cvx::Box>(m_hio.grid()->tree() );
}

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

void WldWidget::testIntersect(const Ray * incident)
{
	m_intersectCtx.reset(*incident);
	if(!m_hio.grid()) return;
	std::stringstream sst; sst<<incident->m_dir;
	qDebug()<<"interset begin "<<sst.str().c_str();
	KdEngine eng;
	eng.intersect<cvx::Box, KdNode4>(m_hio.grid()->tree(), &m_intersectCtx);
	qDebug()<<"interset end";
	if(m_intersectCtx.m_success) {
		qDebug()<<" hit component "<<m_intersectCtx.m_componentIdx;
		m_voxelTree = m_hio.loadCell(m_intersectCtx.m_componentIdx);
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
	
	if(m_intersectCtx.m_success)
		drawActiveSource(m_intersectCtx.m_leafIdx);
}

void WldWidget::drawActiveSource(const unsigned & iLeaf)
{
	if(!m_hio.grid()) return;
	
	glColor3f(0,.6,.4);
	int start, len;
	m_hio.grid()->tree()->leafPrimStartLength(start, len, iLeaf);
	int i=0;
	for(;i<len;++i) {
		const cvx::Box * c = m_hio.grid()->tree()->getSource(start + i);
		BoundingBox b = c->calculateBBox();
		b.expand(-0.03f);
		getDrawer()->boundingBox(b );
	}
}

void WldWidget::drawVoxel()
{
	if(!m_voxelTree) return;
	glColor3f(.0f,.65f,.45f);
	NTreeDrawer dr;
	dr.drawTree<Voxel>(m_voxelTree);
	getDrawer()->setColor(0.f, .15f, .35f);
	// dr.drawTightBox<Voxel>(m_voxelTree);
	dr.drawSource<Voxel>(m_voxelTree);
}

BoundingBox WldWidget::getFrameBox()
{
	BoundingBox b;
	if(m_hio.grid()) b = m_hio.grid()->tree()->getBBox();
	return b;
}
//:~