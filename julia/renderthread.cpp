#include <QtGui>

#include <math.h>

#include "renderthread.h"

RenderThread::RenderThread(QObject *parent)
    : QThread(parent)
{
    restart = false;
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

void RenderThread::render(QSize resultSize)
{
    QMutexLocker locker(&mutex);

    int w = resultSize.width();
    int h = resultSize.height();

/// round to 16
    int tw = w >> 4;
    if((w & 15) > 0) tw++;
    w = tw << 4;
    
    int th = h >> 4;
    if((h & 15) > 0) th++;
    h = th << 4;
  
	this->resultSize = QSize(w, h);

    if (!isRunning()) {
        start(LowPriority);
    } else {
        restart = true;
        condition.wakeOne();
    }
}

void RenderThread::run()
{
    forever {
        mutex.lock();

        QSize resultSize = this->resultSize;

        mutex.unlock();

        QImage image(resultSize, QImage::Format_RGB32);
			
        const int tw = resultSize.width() >> 4;
        const int th = resultSize.height() >> 4;
        
        int i, j, k, l;
        for(j=0; j<th; ++j) {
            for(i=0; i<tw; ++i) {
                if (restart)
                    break;
                if (abort)
                    return;
                
                int r = rand()%256;
                int g = rand()%256;
                int b = rand()%256;
                uint tile[256];
                for(k=0; k<16; ++k) {
                    for(l=0; l<16; ++l) {
                        tile[k * 16 +l] = qRgb(r, g, b);
                    }
                }
                
                for(k=0; k<16; ++k) {
                    uint *scanLine = reinterpret_cast<uint *>(image.scanLine(j * 16 + k));
                    memcpy ( &scanLine[i*16], &tile[k*16], 64 );
                }
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

