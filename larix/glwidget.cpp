#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include "LarixWorld.h"
#include "LarixInterface.h"
#include "AWorldThread.h"

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_world = new LarixWorld;
    m_thread = new AWorldThread(m_world, this);
	LarixInterface::CreateWorld(m_world);
}

GLWidget::~GLWidget()
{
	delete m_world;
}

void GLWidget::clientInit()
{
	connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
    connect(this, SIGNAL(updatePhysics()), m_thread, SLOT(simulate()));
}

void GLWidget::clientDraw()
{
	LarixInterface::DrawWorld(m_world, getDrawer());
    if(m_world->hasSourceP()) {
        std::stringstream sst;
        sst<<"cache range: ("<<m_world->cacheRangeMin()
        <<","<<m_world->cacheRangeMax()
        <<")";
        hudText(sst.str(), 1);
        sst.str("");
        sst<<"current frame: "<<m_world->currentCacheFrame();
        hudText(sst.str(), 2);
        emit updatePhysics();
    }
}

void GLWidget::clientSelect(QMouseEvent * event)
{
	setUpdatesEnabled(false);

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

	setUpdatesEnabled(true);
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	Base3DView::keyPressEvent(e);
}
