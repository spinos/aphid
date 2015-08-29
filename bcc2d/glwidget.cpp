#include <QtGui>
#include "BccInterface.h"
#include <gl_heads.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include "BccWorld.h"

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_interface = new BccInterface;
	m_world = new BccWorld;
	m_interface->createWorld(m_world);
}

GLWidget::~GLWidget()
{
	delete m_interface;
}

void GLWidget::clientInit()
{
	connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
	m_interface->drawWorld(m_world, getDrawer());
	std::stringstream sst;
	sst.str("");
	sst<<"n curves: "<<m_world->numCurves();
	hudText(sst.str(), 1);
	sst.str("");
	sst<<"n tetrahedrons: "<<m_world->numTetrahedrons();
    hudText(sst.str(), 2);
	sst.str("");
	sst<<"n tetrahedron mesh points: "<<m_world->numPoints();
    hudText(sst.str(), 3);
	sst.str("");
	sst<<"n grow mesh triangles: "<<m_world->numTriangles();
    hudText(sst.str(), 4);
}

void GLWidget::clientSelect(QMouseEvent * event)
{
	setUpdatesEnabled(false);
	m_world->select(getIncidentRay());
	setUpdatesEnabled(true);
}

void GLWidget::clientDeselect(QMouseEvent * event) 
{
	setUpdatesEnabled(false);
	
	setUpdatesEnabled(true);
}

void GLWidget::clientMouseInput(QMouseEvent * event)
{
	setUpdatesEnabled(false);
	m_world->select(getIncidentRay());
	setUpdatesEnabled(true);
}

void GLWidget::keyPressEvent(QKeyEvent *event)
{
	if(event->modifiers() == Qt::ControlModifier | Qt::MetaModifier) {
		if(event->key() == Qt::Key_S) {
			if(preSaveCheck()) m_interface->saveWorld(m_world);
		}
	}
    setUpdatesEnabled(false);	
	switch (event->key()) {
		case Qt::Key_K:
		    m_world->rebuildTetrahedronsMesh(-1.f);
			break;
		case Qt::Key_L:
		    m_world->rebuildTetrahedronsMesh(1.f);
			break;
		case Qt::Key_M:
		    m_world->reduceSelected(.13f);
			break;
		/*case Qt::Key_S:
		    m_world->moveTestP(0.f, -.1f, 0.f);
			break;
		case Qt::Key_F:
		    m_world->moveTestP(0.f, 0.f, .1f);
			break;
		case Qt::Key_B:
		    m_world->moveTestP(0.f, 0.f, -.1f);
			break;*/
		default:
			break;
	}
	setUpdatesEnabled(true);
	Base3DView::keyPressEvent(event);
}

void GLWidget::importGrowMesh()
{
	QString selectedFilter;
	QString fileName = QFileDialog::getOpenFileName(this,
							tr("Open .hes file for grow mesh"),
							tr("info"),
							tr("All Files (*);;Hesperis Files (*.hes)"),
							&selectedFilter,
							QFileDialog::DontUseNativeDialog);
	if(fileName == "") return;
    if(!m_interface->loadTriangleGeometry(m_world, fileName.toUtf8().data()))
        qDebug()<<" failed to load grow mesh from hes file: "<<fileName;
}

void GLWidget::importCurve()
{
	QString selectedFilter;
	QString fileName = QFileDialog::getOpenFileName(this,
							tr("Open .hes file for curve"),
							tr("info"),
							tr("All Files (*);;Hesperis Files (*.hes)"),
							&selectedFilter,
							QFileDialog::DontUseNativeDialog);
	if(fileName == "") return;
    if(m_interface->loadCurveGeometry(m_world, fileName.toUtf8().data()))
		m_world->buildTetrahedronMesh();
	else
		qDebug()<<" failed to load curve from hes file: "<<fileName;
}

void GLWidget::importPatch()
{
	QString selectedFilter;
	QString fileName = QFileDialog::getOpenFileName(this,
							tr("Open .hes file for triangle patch"),
							tr("info"),
							tr("All Files (*);;Hesperis Files (*.hes)"),
							&selectedFilter,
							QFileDialog::DontUseNativeDialog);
	if(fileName == "") return;
    if(m_interface->loadPatchGeometry(m_world, fileName.toUtf8().data())) {
	
	}
	else
		qDebug()<<" failed to load patch from hes file: "<<fileName;
}

bool GLWidget::preSaveCheck()
{
	if(m_interface->FileName == "unknown") {
		QString selectedFilter;
		QString fileName = QFileDialog::getSaveFileName(this,
							tr("Save current scene to .hes file"),
							tr("info"),
							tr("All Files (*);;Hesperis Files (*.hes)"),
							&selectedFilter,
							QFileDialog::DontUseNativeDialog);
		if(fileName == "") return false;
		
		m_interface->FileName = fileName.toUtf8().data();
	}
	return true;
}
//:~