#include <QtGui>
#include <iostream>
#include "LfThread.h"
#include "LfWorld.h"
#include <ExrImage.h>

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
	
	ExrImage img;
	const int n = m_world->param()->numImages();
	int i, j;
	
	uint *spasityLine = reinterpret_cast<uint *>(m_spasityImg->bits());
	
	unsigned cwhite = 255<<24;
	cwhite = cwhite | ( 255 << 16 );
	cwhite = cwhite | ( 255 << 8 );
	cwhite = cwhite | ( 255 );
	
	const int totalNSignals = m_world->param()->totalNumPatches();
	int niter = 0;
	int t = 0;
	int endp;

	forever {
	    int ns = 0;
		for(i=0;i<n;i++) {
			img.open(m_world->param()->imageName(i));
			const int m = m_world->param()->imageNumPatches(i);
			int nbatch = m>>8;
			if( (nbatch<<8) < m ) nbatch++;
			for(j=0;j<nbatch;j++) {
			    endp = (j+1) * 256 - 1;
			    if(endp > m-1) endp = m-1;
				m_world->learn(&img, j * 256, endp);
				
				{
/// force to clean on first batch of first image only
                    // if( i==0 && j==0 ) 
                    // qDebug()<<" t"<<t<<" nit"<<niter;
				    m_world->updateDictionary( &img, niter );
					m_world->dictionaryAsImage(scanLine, w, h);
					emit sendDictionary(*m_dictImg);
					emit sendSparsity(*m_spasityImg);
				}
			}
			ns += m;
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
        
        niter++;
        //m_world->cleanDictionary();
        //if(niter > 1) 
        {
            m_world->recycleData();
            qDebug()<<" recycle"<<niter;
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
