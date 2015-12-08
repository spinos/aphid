#include <QtGui>
#include <iostream>
#include "LfThread.h"
#include "LfMachine.h"
#include <ExrImage.h>

namespace lfr {
LfThread::LfThread(LfMachine * world, QObject *parent)
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
    endLearn();
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

void LfThread::endLearn()
{
	if(isRunning() ) {
	mutex.lock();
    abort = true;
    condition.wakeOne();
    mutex.unlock();

    wait();
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
	QElapsedTimer timer;
	timer.start();
	int endp;
	int niter = 0;
	for(;niter< m_world->param()->maxIterations();++niter) {
		for(i=0;i<n;i++) {
			mutex.lock();
			if (abort) return;
			mutex.unlock();
			
			img.open(m_world->param()->imageName(i));
			const int m = m_world->param()->imageNumPatches(i);
			int nbatch = m>>8;
			if( (nbatch<<8) < m ) nbatch++;
			for(j=0;j<nbatch;j++) {
			    endp = (j+1) * 256 - 1;
			    if(endp > m-1) endp = m-1;
				m_world->learn(&img, j * 256, endp);
				
				{
				    m_world->updateDictionary( &img, niter );
					m_world->dictionaryAsImage(scanLine, w, h);
					emit sendDictionary(*m_dictImg);
					// emit sendSparsity(*m_spasityImg);
				}
			}
		}
		
		for(i=0;i<n;i++) {
			mutex.lock();
			if (abort) return;
			mutex.unlock();
			
			img.open(m_world->param()->imageName(i));
#if 1
            float e = m_world->computePSNR(&img, i);
            emit sendPSNR(e);
#else
            const int m = m_world->param()->imageNumPatches(i);
			m_world->beginPSNR();
			for(j=0;j<m;j++) {
				m_world->computeError(&img, j);
				//m_world->fillSparsityGraph(spasityLine, j & 255, 100, cwhite);
				
				//if(((j+1) & 255) == 0 || (j+1)==m) {
				//	emit sendSparsity(*m_spasityImg);
				//}
			}
			float err;
			m_world->endPSNR(&err);
			emit sendPSNR(err);
#endif
			
		}
        
        //m_world->cleanDictionary();
        //if(niter > 1) 
        {
            m_world->recycleData();
			emit sendIterDone(niter+1);
			// timer.elapsed() << "milliseconds";
            //qDebug()<<" recycle"<<niter;
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
