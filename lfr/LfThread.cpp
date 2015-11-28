#include <QtGui>
//#include <cmath>
#include <iostream>
#include "LfThread.h"
#include "LfWorld.h"

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
	int w, h;
	m_world->param()->getDictionaryImageSize(w, h);
	m_world->initDictionary();
	QImage img(w, h, QImage::Format_RGB32);
	uint *scanLine = reinterpret_cast<uint *>(img.bits());
	m_world->dictionaryAsImage(scanLine, w, h);
	emit sendInitialDictionary(img);
}

void LfThread::beginLearn()
{
	m_world->preLearn();
	
	QMutexLocker locker(&mutex);

	if (!isRunning()) {
        start(LowPriority);
    } else {
        restart = true;
        condition.wakeOne();
    }
}

void LfThread::run()
{
	int w, h;
	m_world->param()->getDictionaryImageSize(w, h);
	QImage img(w, h, QImage::Format_RGB32);
	uint *scanLine = reinterpret_cast<uint *>(img.bits());
		
	int i=0;
	for(;i<200;i++) {
		m_world->learn(0, i);
		
		if((i & 3) == 0) {
			m_world->dictionaryAsImage(scanLine, w, h);
			emit sendDictionary(img);
		}
	}
/*
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
					int g = rand()%256;
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
*/
}
}
