#include <QtGui>
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
/// 128 x 256 spasity visualization
	m_spasityImg = new QImage(128, 256, QImage::Format_RGB32);
	int w, h;
	world->param()->getDictionaryImageSize(w, h);
	m_dictImg = new QImage(w, h, QImage::Format_RGB32);
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
	uint *scanLine = reinterpret_cast<uint *>(m_dictImg->bits());
		
	const int n = m_world->param()->imageNumPatches(0);
	uint *spasityLine = reinterpret_cast<uint *>(m_spasityImg->bits());
	
	unsigned cwhite = 255<<24;
	cwhite = cwhite | ( 255 << 16 );
	cwhite = cwhite | ( 255 << 8 );
	cwhite = cwhite | ( 255 );
	
	forever {
	int i=0;
	for(;i<n;i++) {
		m_world->learn(0, i);
		m_world->fillSparsityGraph(spasityLine, i & 255, 128, cwhite);
		
		if(((i+1) & 255) == 0 || (i+1)==n) {
			m_world->updateDictionary();
			m_world->dictionaryAsImage(scanLine, w, h);
			emit sendDictionary(*m_dictImg);
			emit sendSparsity(*m_spasityImg);
		}
	}
		float err;
		m_world->computePSNR(&err, 0);
		std::cout<<"\n PSNR "<<err;
		std::cout<<"\n repeat";
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
