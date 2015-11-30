#include <QtGui>
#include <iostream>
#include "LfThread.h"
#include "LfWorld.h"
#include <zEXRImage.h>

namespace lfr {
LfThread::LfThread(LfWorld * world, QObject *parent)
    : QThread(parent)
{
	m_world = world;
    restart = false;
    abort = false;
/// 100 x 256 spasity visualization
	m_spasityImg = new QImage(100, 256, QImage::Format_RGB32);
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
		
	
	ZEXRImage img;
	const int n = m_world->param()->numImages();
	int i, j;
	
	uint *spasityLine = reinterpret_cast<uint *>(m_spasityImg->bits());
	
	unsigned cwhite = 255<<24;
	cwhite = cwhite | ( 255 << 16 );
	cwhite = cwhite | ( 255 << 8 );
	cwhite = cwhite | ( 255 );
	
	forever {
		for(i=0;i<n;i++) {
			img.open(m_world->param()->imageName(i));
			const int m = m_world->param()->imageNumPatches(i);
			for(j=0;j<m;j++) {
				m_world->learn(&img, j);
				m_world->fillSparsityGraph(spasityLine, j & 255, 100, cwhite);
			
				if(((j+1) & 255) == 0 || (j+1)==m) {
					m_world->updateDictionary();
					m_world->dictionaryAsImage(scanLine, w, h);
					emit sendDictionary(*m_dictImg);
					emit sendSparsity(*m_spasityImg);
				}
			}
		}
		
		for(i=0;i<n;i++) {
			img.open(m_world->param()->imageName(i));
			const int m = m_world->param()->imageNumPatches(i);
			m_world->beginPSNR();
			for(j=0;j<m;j++) {
				m_world->computeError(&img, j);
				m_world->fillSparsityGraph(spasityLine, j & 255, 100, cwhite);
				
				if(((j+1) & 255) == 0 || (j+1)==m) {
					emit sendSparsity(*m_spasityImg);
				}
			}
			float err;
			m_world->endPSNR(&err);
			emit sendPSNR(err);
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
