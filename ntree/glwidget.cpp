#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <GeoDrawer.h>
#include "NTreeDrawer.h"
#include <NTreeIO.h>
#include <HWorldGrid.h>
#include <HInnerGrid.h>
#include "GridDrawer.h"

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
	m_grid = NULL;
	
	std::cout<<"\n sizeof 4node "<<sizeof(KdNode4);
	if(filename.size() > 1) readTree(filename);
	else {
		testTree();
		// testGrid();
	}
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
	delete m_source;
	delete m_tree;
	if(m_grid) delete m_grid;
}

void GLWidget::testTree()
{
	std::cout<<"\n test kdtree";
	const int n = 27913;
    m_source = new sdb::VectorArray<cvx::Cube>();
	m_tree = new KdNTree<cvx::Cube, KdNode4 >();
	
	BoundingBox rootBox;
    int i;
    for(i=0; i<n; i++) {
        cvx::Cube a;
        float r = sqrt(float( rand() % 999 ) / 999.f);
        float th = float( rand() % 999 ) / 999.f * 1.5f;
        float x = -60.f + 150.f * r * cos(th*1.1f);
        float y = 0.f + 80.f * r * sin(th/.93f) + 39.f * sin(x/13.f);
        float z = 0.f + 60.f * float( rand() % 999 ) / 999.f + 24.f * sin(y/23.f);
        a.set(Vector3F(x, y, z), .2f);
        
		m_source->insert(a);
		rootBox.expandBy(a.calculateBBox());
    }
	
	std::cout<<"\n bbox "<<rootBox;
	rootBox.round();
	std::cout<<"\n rounded to "<<rootBox;
	
    TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 32;
	
	KdEngine engine;
    engine.buildTree<cvx::Cube, KdNode4, 4>(m_tree, m_source, rootBox, &bf);
	// engine.printTree<cvx::Cube, KdNode4>(m_tree);
}

void GLWidget::clientInit()
{
	connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
	drawBoxes();
	drawTree();
	drawIntersect();
	// drawGrid();
}

void GLWidget::drawBoxes() const
{
	if(!m_source) return;
    getDrawer()->setColor(.065f, .165f, .065f);
#if 0
    const int n = m_source->size();
    int i = 0;
    for(;i<n;i++) {
		BoundingBox b = m_source->get(i)->calculateBBox();
		b.expand(-0.03f);
        getDrawer()->boundingBox(b );
    }
#else
	NTreeDrawer dr;
	dr.drawSource<cvx::Cube>(m_tree);
#endif
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
		case Qt::Key_F:
			camera()->frameAll(getFrameBox() );
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
	
	if(m_intersectCtx.m_success) drawActiveSource(m_intersectCtx.m_leafIdx);
}

void GLWidget::drawActiveSource(const unsigned & iLeaf)
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

void GLWidget::testGrid()
{
	m_grid = new VoxelGrid<KdNTree<cvx::Cube, KdNode4 >, cvx::Cube >();
	BoundingBox b = m_tree->getBBox();
	b.expand(b.getLongestDistance() * .005f);
	m_grid->create(m_tree, b);
	
	m_source->clear();
	m_grid->extractCellBoxes(m_source);
	
	BoundingBox rootBox;
	m_grid->getBounding(rootBox);
	
	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 16;
    
	KdEngine engine;
    engine.buildTree<cvx::Cube, KdNode4, 4>(m_tree, m_source, rootBox, &bf);
}

void GLWidget::drawGrid()
{
	if(!m_grid) return;
	
	glColor3f(0,.3,.4);
	GridDrawer dr;
	dr.drawGrid<CartesianGrid>(m_grid);
}

BoundingBox GLWidget::getFrameBox()
{
	BoundingBox b;
	if(m_tree) b = m_tree->getBBox();
	return b;
}
//:~