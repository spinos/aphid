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
	m_numLoops = 0;
}

AWorldThread::~AWorldThread()
{
    mutex.lock();
    abort = true;
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

void AWorldThread::run()
{	 
    forever 
    {
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

        m_world->progressFrame();
        
        m_numLoops += NumSubsteps; 

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
//:~
