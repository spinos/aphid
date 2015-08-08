#include <QtCore>
#include "AWorldThread.h"
#include "AWorld.h"
float AWorldThread::TimeStep = 1.f / 60.f;
int AWorldThread::NumSubsteps = 2;
AWorldThread::AWorldThread(AWorld * world, QObject *parent)
    : QThread(parent)
{
    m_world = world;
    abort = false;
	restart = false;
    blockTransfer = true;
	m_numLoops = 0;
}

AWorldThread::~AWorldThread()
{
    mutex.lock();
    abort = true;
    blockTransfer = false;
    condition.wakeOne();
    mutex.unlock();
    wait();
}

void AWorldThread::simulate()
{
    QMutexLocker locker(&mutex);

    if (!isRunning()) {
        start(LowPriority);
    } else {
		restart = true;
        condition.wakeOne();
    }
}

void AWorldThread::lockTransfer(bool x)
{ 
    QMutexLocker locker(&mutex);

    blockTransfer = x;

    qDebug()<<"set lockTransfer "<<blockTransfer;
}

void AWorldThread::run()
{	 
    //forever 
    {
        //bool waitTransfer = blockTransfer;
        //qDebug()<<"get lockTransfer "<<waitTransfer;
        //if(waitTransfer) {
        //    qDebug()<<"wait for transfer lock";
        //    condition.wait(&mutex);
       // }
        
        if (abort) {
            qDebug()<<"abort physics b4";
            return;
        }
		
		m_world->prePhysics();
        
        for(int i=0; i < NumSubsteps; i++) {
           m_world->stepPhysics(TimeStep);
        }
        
        if (abort) {
            qDebug()<<"abort physics aft";
            return;
        }
		
		m_world->postPhysics();

        //mutex.lock();
        m_world->progressFrame();
        //mutex.unlock(); 
        
        m_numLoops+=NumSubsteps; 

        qDebug()<<"done step";
		emit doneStep();

		mutex.lock();
        
        if (!restart)
			condition.wait(&mutex);
			
		restart = false;

        mutex.unlock(); 
    }
}

const unsigned AWorldThread::numLoops() const
{ return m_numLoops; }

void AWorldThread::block()
{ mutex.lock(); }

void AWorldThread::unblock()
{ mutex.unlock(); }
//:~
