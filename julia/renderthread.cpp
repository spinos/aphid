#include <QtGui>

#include <math.h>

#include "renderthread.h"

RenderThread::RenderThread(QObject *parent)
    : QThread(parent)
{
    restart = false;
    abort = false;
	m_r = NULL;
}

RenderThread::~RenderThread()
{
    mutex.lock();
    abort = true;
    condition.wakeOne();
    mutex.unlock();

    wait();
}

void RenderThread::setR(aphid::CudaRender * r)
{ m_r = r; }

void RenderThread::render(QSize resultSize)
{
    QMutexLocker locker(&mutex);

    this->m_portSize = resultSize;
    
    int w = resultSize.width();
    int h = resultSize.height();

	m_r->getRoundedSize(w, h);
	
	this->m_resultSize = QSize(w, h);

    if (!isRunning()) {
        start(LowPriority);
    } else {
        restart = true;
        condition.wakeOne();
    }
}

void RenderThread::tumble(int dx, int dy)
{
	QMutexLocker locker(&mutex);
	int w = this->m_portSize.width();
	m_r->tumble(dx, dy, w);
	
	if (!isRunning()) {
		start(LowPriority);
    } else {
		restart = true;
        condition.wakeOne();
    }
}

void RenderThread::track(int dx, int dy)
{
	QMutexLocker locker(&mutex);
	int w = this->m_portSize.width();
	m_r->track(dx, dy, w);
	
	if (!isRunning()) {
        start(LowPriority);
    } else {
        restart = true;
        condition.wakeOne();
    }
}

void RenderThread::zoom(int dz)
{
	QMutexLocker locker(&mutex);
	int w = this->m_portSize.width();
	m_r->zoom(dz, w);
	
	if (!isRunning()) {
        start(LowPriority);
    } else {
        restart = true;
        condition.wakeOne();
    }
}

void RenderThread::frameAll()
{
    QMutexLocker locker(&mutex);
	
    m_r->frameAll();
	
	if (!isRunning()) {
        start(LowPriority);
    } else {
        restart = true;
        condition.wakeOne();
    }
}

void RenderThread::run()
{
	int nrender = 0;
   for(;;) {
        mutex.lock();

        QSize renderSize = this->m_resultSize;
		m_r->setPortSize(m_portSize.width(), m_portSize.height() );
		m_r->setBufferSize(renderSize.width(), renderSize.height() );
		m_r->updateRayFrameVec();
	
        mutex.unlock();
		
		if (abort) {
			qDebug()<<"abort render loop"<<++nrender;
			return;
		}
		
		if (!restart) {	
		    qDebug()<<"render loop"<<++nrender;
		    m_r->render();

		//qDebug()<<" imagesize "<<renderSize.width()<<"x"<<renderSize.height();
		
		    QImage image(renderSize, QImage::Format_RGB32);
		    m_r->colorToHost(reinterpret_cast<uint *>(image.scanLine(0) ), 
		                    renderSize.width() * renderSize.height() );
		
		    emit renderedImage(image);
		}
			
        mutex.lock();
		
        if (!restart)
            condition.wait(&mutex);
			
        restart = false;
        mutex.unlock();
    }
}

