#include <QtGui>
#include <iostream>
#include "LfThread.h"
#include "LfMachine.h"
#include <ExrImage.h>

using namespace aphid;

namespace lfr {
LfThread::LfThread(LfMachine * world, QObject *parent)
    : QThread(parent)
{
	m_world = world;
    restart = false;
    abort = false;
/// 100 x 256 spasity visualization
	m_spasityImg = new QImage(100, 256, QImage::Format_RGB32);
	int dw = 10, dh = 10, w = 10, h = 10;
	const LfParameter * lparam = world->param();
	if(lparam->isValid() ) {
		lparam->getDictionaryImageSize(dw, dh);
		lparam->getImageSize(w, h, 0);
	}
	m_dictImg = new QImage(dw, dh, QImage::Format_RGB32);
	m_codedImg = new QImage(w, h, QImage::Format_RGB32);
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
	const LfParameter * lparam = m_world->param();
	int w, h;
	lparam->getDictionaryImageSize(w, h);
	uint *scanLine = reinterpret_cast<uint *>(m_dictImg->bits());
	
	ExrImage img;
	const int n = lparam->numImages();
	int i, j;
	
	uint *spasityLine = reinterpret_cast<uint *>(m_spasityImg->bits());
	
#if 0
	unsigned cwhite = 255<<24;
	cwhite = cwhite | ( 255 << 16 );
	cwhite = cwhite | ( 255 << 8 );
	cwhite = cwhite | ( 255 );
#endif

	const int totalNSignals = lparam->totalNumPatches();
	QElapsedTimer timer;
	timer.start();
	int endp;
	int niter = 0;
	for(;niter< lparam->maxIterations();++niter) {
		for(i=0;i<n;i++) {
			mutex.lock();
			if (abort) return;
			mutex.unlock();
			
			img.read(lparam->imageName(i) );
			const int m = lparam->imageNumPatches(i);
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
			
			img.read(lparam->imageName(i));
#if 1
            float e = m_world->computePSNR(&img, i);
            emit sendPSNR(e);
			
			if(i==0) {
/// reconstruct image
				uint *codedLine = reinterpret_cast<uint *>(m_codedImg->bits());
				m_world->computeYhat(codedLine, i, &img);
			}
#else
            const int m = lparam->imageNumPatches(i);
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

}
}
