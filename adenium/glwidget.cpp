#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include "AdeniumInterface.h"
#include "AdeniumWorld.h"
#include <WorldDbgDraw.h>

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(10000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(10000.f);
	orthoCamera()->setNearClipPlane(1.f);
    connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
    m_world = new AdeniumWorld;
    AdeniumInterface adei;
    adei.create(m_world);
	m_enableRayCast = true;
}

GLWidget::~GLWidget()
{
    delete m_world;
}

void GLWidget::clientInit()
{	
	AdeniumWorld::DbgDrawer = new WorldDbgDraw(getDrawer());
    m_world->initOnDevice();
}

void GLWidget::clientDraw()
{
	if(m_enableRayCast) {
		m_world->render(getCamera());
		glDisable(GL_BLEND);
		drawFrontImagePlane();
	}
	else {
		m_world->draw(getCamera());
	}
		
	std::stringstream sst;
	sst.str("");
	sst<<"fps: "<<frameRate();
	hudText(sst.str(), 1);
}
//! [7]

//! [9]
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
	AdeniumInterface adei;
	switch (e->key()) {
		case Qt::Key_Q:
			toggleRayCast();
			break;
        case Qt::Key_L:
			loadBake();
			break;
        case Qt::Key_A:
			adei.changeMaxDisplayLevel(m_world, 1);
			break;
		case Qt::Key_D:
			adei.changeMaxDisplayLevel(m_world, -1);
			break;
        default:
			break;
    }
	Base3DView::keyPressEvent(e);
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base3DView::keyReleaseEvent(event);
}

void GLWidget::resizeEvent(QResizeEvent * event)
{
	QSize renderAreaSize = size();
    //qDebug()<<"render size "<<renderAreaSize.width()<<" "<<renderAreaSize.height();
    m_world->resizeRenderArea(renderAreaSize.width(), renderAreaSize.height());
    Base3DView::resizeEvent(event);
}

void GLWidget::toggleRayCast()
{
	if(m_enableRayCast) m_enableRayCast = false;
	else m_enableRayCast = true;
}

void GLWidget::loadBake()
{
    QString selectedFilter;
	QString fileName = QFileDialog::getOpenFileName(this,
							tr("Open .h5 file as baked grow mesh"),
							tr("info"),
							tr("All Files (*);;H5 Files (*.h5)"),
							&selectedFilter,
							QFileDialog::DontUseNativeDialog);
	if(fileName == "") return;
    if(!AdeniumInterface::LoadBake(m_world, fileName.toUtf8().data()))
        qDebug()<<" failed to load h5 bake file: "<<fileName;
}
//:~