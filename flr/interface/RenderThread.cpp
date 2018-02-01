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
    m_abort = false;
}

RenderThread::~RenderThread()
{
    interruptRender();
}

void RenderThread::interruptRender()
{
	if(isRunning() ) {
		mutex.lock();
		this->m_abort = true;
		condition.wakeOne();
		mutex.unlock();

		wait();
	}
}

void RenderThread::interruptAndResize()
{
//	qDebug()<<"interruptAndResize";	
	interruptRender();
	
	m_interface->createImage(m_interface->resizedImageWidth(),
							m_interface->resizedImageHeight() );
	
	m_interface->updateDisplayView();
	
	this->m_abort = false;
	start(LowPriority);
}

void RenderThread::interruptAndReview()
{
//	qDebug()<<"interruptAndReview";
	interruptRender();
	
	m_interface->updateDisplayView();
	
	this->m_abort = false;
	start(LowPriority);
}

void RenderThread::render()
{
	if(m_interface->imageSizeChanged() ) {
		interruptAndResize();
		return;
	}
	
	if(m_interface->cameraChanged() ) {
		interruptAndReview();
		return;
	}

	QMutexLocker locker(&mutex);
	
	this->m_abort = false;
	
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
        
        mutex.unlock();
				
		if (m_abort) {
			//qDebug()<<" abort";
			return;
		}
		
		if(m_interface->isResidualLowEnough() ) {
			return;
		}
		
		BufferBlock* packet = m_interface->selectBlock();
		Renderer* tracer = m_interface->getRenderer();
		RenderContext* ctx = m_interface->getContext();
		
		tracer->renderFragment(*ctx, *packet);
					
		packet->projectImage(m_interface->image() );

		emit renderedImage();

        mutex.lock();
       // if (!restart)
		//condition.wait(&mutex);
			
        //restart = false;
        mutex.unlock();
    }
}
