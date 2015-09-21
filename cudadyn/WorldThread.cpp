#include <QtCore>
#include "WorldThread.h"

float WorldThread::TimeStep = 1.f / 60.f;
int WorldThread::NumSubsteps = 2;
WorldThread::WorldThread(CudaDynamicWorld * world, QObject *parent)
    : QThread(parent)
{
    m_world = world;
    abort = false;
	restart = false;
	m_numLoops = 0;
}

WorldThread::~WorldThread()
{
    mutex.lock();
    abort = true;
    condition.wakeOne();
    mutex.unlock();
    wait();
}

void WorldThread::simulate()
{
    QMutexLocker locker(&mutex);

    if (!isRunning()) {
        start(LowPriority);
    } else {
		restart = true;
        condition.wakeOne();
    }
}

void WorldThread::run()
{	 
    forever 
    {
        if (abort) {
            qDebug()<<"abort physics b4";
            return;
        }
        
        for(int i=0; i < NumSubsteps; i++) {
           m_world->stepPhysics(TimeStep);
        }
        
        if (abort) {
            qDebug()<<"abort physics aft";
            return;
        }
        
       m_world->updateEnergy();
       // m_world->putToSleep();
// before step?
        m_world->readVelocityCache();
        
        m_world->sendXToHost();
        m_world->saveCache();
        m_world->reset();
        
        m_numLoops+=NumSubsteps; 

		emit doneStep();

		mutex.lock();
        
        if (!restart)
			condition.wait(&mutex);
			
		restart = false;

        mutex.unlock(); 
    }
}

const unsigned WorldThread::numLoops() const
{ return m_numLoops; }

