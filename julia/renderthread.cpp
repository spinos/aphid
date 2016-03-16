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

void RenderThread::run()
{
    for(;;) {
        mutex.lock();

        QSize renderSize = this->m_resultSize;

        mutex.unlock();
		
		if (restart)
			break;
		if (abort)
			return;
		
		m_r->setPortSize(m_portSize.width(), m_portSize.height() );
		m_r->setBufferSize(renderSize.width(), renderSize.height() );
		m_r->render();
		
		QImage image(renderSize, QImage::Format_RGB32);
			
        const int & tw = m_r->tileX();
        const int & th = m_r->tileY();
		const int & ts = m_r->tileSize();
		
		// qDebug()<<" tile "<<tw<<"x"<<th<<" size"<<ts;
        
        int i, j, k, l;
        for(j=0; j<th; ++j) {
            for(i=0; i<tw; ++i) {
                if (restart)
                    break;
                if (abort)
                    return;
                
				uint * tile = (uint *)m_r->tileHostColor(i, j);
                
				uint *scanLine = reinterpret_cast<uint *>(image.scanLine(j * ts) );
				m_r->sendTileColor(&scanLine[i*ts], renderSize.width(), i, j);
            }
        }
        
		if (!restart)
			emit renderedImage(image);

        mutex.lock();
		
        if (!restart)
            condition.wait(&mutex);
			
        restart = false;
        mutex.unlock();
    }
}

