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
    m_interface = new LarixInterface;
	m_world = new LarixWorld;
    m_interface->createWorld(m_world);
    m_thread = new AWorldThread(m_world, this);
}

GLWidget::~GLWidget()
{
	delete m_thread;
    delete m_world;
    delete m_interface;
}

void GLWidget::clientInit()
{
	// connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
    //connect(this, SIGNAL(updatePhysics()), m_thread, SLOT(simulate()));
    //connect(m_thread, SIGNAL(doneStep()), this, SLOT(update()));
    //connect(this, SIGNAL(isDrawing(bool)), m_thread, SLOT(lockTransfer(bool)));
    //emit updatePhysics();
}

void GLWidget::clientDraw()
{
    m_world->progressFrame();
	m_interface->drawWorld(m_world, getDrawer());

    if(m_world->hasSourceP()) {
        std::stringstream sst;
        sst<<"cache range: ("<<m_world->cacheRangeMin()
        <<","<<m_world->cacheRangeMax()
        <<")";
        hudText(sst.str(), 1);
        sst.str("");
        sst<<"current frame: "<<m_world->currentCacheFrame();
        hudText(sst.str(), 2);
        //emit updatePhysics();
        
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

void GLWidget::keyPressEvent(QKeyEvent * event)
{
	switch (event->key()) {
        case Qt::Key_S:
			saveField();
            break;
        default:
			break;
    }
	
	Base3DView::keyPressEvent(event);
}

void GLWidget::saveField()
{
	QString selectedFilter;
	QString fileName = QFileDialog::getSaveFileName(this,
							tr("Save .fld file as output dP field"),
							tr("info"),
							tr("All Files (*);;Field Files (*.fld)"),
							&selectedFilter,
							QFileDialog::DontUseNativeDialog);
	if(fileName == "") return;
    if(!m_world->setFileOut(fileName.toUtf8().data()))
        qDebug()<<" failed to save field file: "<<fileName;
}
