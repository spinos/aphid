#include <QtGui>

#include <math.h>

#include "renderthread.h"
#include "RenderInterface.h"
#include "BufferBlock.h"
#include "NoiseRenderer.h"

RenderThread::RenderThread(RenderInterface* interf, QObject *parent)
    : QThread(parent)
{
	m_interface = interf;
    abort = false;
}

RenderThread::~RenderThread()
{
    mutex.lock();
	abort = true;
    condition.wakeOne();
    mutex.unlock();

    wait();
}

void RenderThread::render(double centerX, double centerY, double scaleFactor,
                          QSize resultSize)
{
    QMutexLocker locker(&mutex);
	
	this->m_resultSize = resultSize;
    this->centerX = centerX;
    this->centerY = centerY;
    this->scaleFactor = scaleFactor;
    this->abort = false;
	
    if (!isRunning()) {
        start(LowPriority);
    } else {
        condition.wakeOne();
    }
}

void RenderThread::run()
{
    forever {
		
        mutex.lock();
        QSize resultSize = this->m_resultSize;
		
        mutex.unlock();
				
		if (abort) {
			qDebug()<<" abort";
			return;
		}
		
		if(m_interface->imageSizeChanged(resultSize.width(), resultSize.height() ) ) {
			qDebug()<<" size changed "<<resultSize;
			m_interface->createImage(resultSize.width(), resultSize.height() );
			
		}
		
		BufferBlock* packet = m_interface->selectABlock(m_interface->bufferNumBlocks() );
		Renderer* tracer = m_interface->getRenderer();
		
		tracer->traceRays(*packet);
		delete tracer;
					
		packet->projectImage(m_interface->image() );

		emit renderedImage();

        mutex.lock();
       // if (!restart)
		//condition.wait(&mutex);
			
        //restart = false;
        mutex.unlock();
    }
}
