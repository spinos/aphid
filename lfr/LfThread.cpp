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
	uint *codedLine = reinterpret_cast<uint *>(m_codedImg->bits());
				
	ExrImage img;
	const int nimage = lparam->numImages();
	int i, j, m;
	float e;
	
	const int totalNSignals = lparam->totalNumPatches();
	QElapsedTimer timer;
	timer.start();

	srand(1);
	int niter = 0;
	for(;niter< lparam->maxIterations();++niter) {
		
		for(i=0;i<nimage;i++) {
			mutex.lock();
			if (abort) return;
			mutex.unlock();
			
			img.read(lparam->imageName(i) );
			m = lparam->imageNumPatches(i);
			
			m_world->learn(&img, m);
			m_world->updateDictionary( &img, niter );
			
			m_world->dictionaryAsImage(scanLine, w, h);
			emit sendDictionary(*m_dictImg);

		}
		
		for(i=0;i<nimage;i++) {
			mutex.lock();
			if (abort) return;
			mutex.unlock();
			
			img.read(lparam->imageName(i));

            e = m_world->computePSNR(&img, i);
            emit sendPSNR(e);
			
			if(i==0) {
				m_world->computeYhat(codedLine, i, &img);
				emit sendCodedImage(*m_codedImg);
				
			}
			
		}
        
        {
            m_world->recycleData();
			emit sendIterDone(niter+1);
			// timer.elapsed() << "milliseconds";
            //qDebug()<<" recycle"<<niter;
        }
		m_world->addLambda();
		
	}

}
}
