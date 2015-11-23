#include <QtGui>
#include <cmath>
#include <iostream>
#include "LfThread.h"
namespace lfr {
LfThread::LfThread(LfWorld * world, QObject *parent)
    : QThread(parent)
{
	m_world = world;
    restart = false;
    abort = false;

}

LfThread::~LfThread()
{
    mutex.lock();
    abort = true;
    condition.wakeOne();
    mutex.unlock();

    wait();
}

void LfThread::render(QSize resultSize)
{
    QMutexLocker locker(&mutex);

	this->resultSize = resultSize;

    if (!isRunning()) {
        start(LowPriority);
    } else {
        restart = true;
        condition.wakeOne();
    }
}

void LfThread::initAtoms()
{
	const int s = m_world->param()->atomSize();
	const int k = m_world->param()->dictionaryLength();
	int w = 8; int h = k / w;
	while(h>w) {
		w<<=1;
		h = k / w;
	}
	if(h*w < k) h++;
	
	QImage img(w*s, h*s, QImage::Format_RGB32);
	uint *scanLine = reinterpret_cast<uint *>(img.bits());
	m_world->fillDictionary(scanLine, w*s, h*s);
	
	emit renderedImage(img);
}

void LfThread::run()
{
    forever {
        mutex.lock();

        QSize resultSize = this->resultSize;

        mutex.unlock();

        QImage image(resultSize, QImage::Format_RGB32);
			
			for (int y = 0; y < resultSize.height(); ++y) 
			{
				 if (restart)
                    break;
                if (abort)
                    return;

                uint *scanLine = reinterpret_cast<uint *>(image.scanLine(y));
				for (int x = 0; x < resultSize.width(); ++x) 
				{
					int g = random()%256;
					*scanLine++ = qRgb(g, g, g);
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
}
