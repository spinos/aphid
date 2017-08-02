#include <QtCore>
#include "BaseSolverThread.h"

namespace aphid {

float BaseSolverThread::TimeStep = 1.f / 30.f;
int BaseSolverThread::NumSubsteps = 9;
BaseSolverThread::BaseSolverThread(QObject *parent)
    : QThread(parent)
{
    abort = false;
	restart = false;
	m_numLoops = 0;
	m_numTicks = 0;
}

BaseSolverThread::~BaseSolverThread()
{
    mutex.lock();
    abort = true;
    condition.wakeOne();
    mutex.unlock();
    wait();
}

void BaseSolverThread::simulate()
{
    QMutexLocker locker(&mutex);

    if (!isRunning()) {
        start(LowPriority);
    } else {
		restart = true;
        condition.wakeOne();
    }
}

void BaseSolverThread::run()
{	 
	forever {
	    if (abort) {
            // destroySolverData();
            qDebug()<<"abort";
            return;
        }
        
		const float dt = TimeStep / NumSubsteps;
        for(int i=0; i < NumSubsteps;++i)
            stepPhysics(dt);
		
		emit doneStep();
		m_numTicks++;
		if(isMakingCache() ) {
		    processMakingCache();
		}

		mutex.lock();
		
        if (!restart)
			condition.wait(&mutex);
			
		restart = false;
        mutex.unlock();
   }
}

void BaseSolverThread::stepPhysics(float dt)
{ m_numLoops++; }

const unsigned BaseSolverThread::numLoops() const
{ return m_numLoops; }

void BaseSolverThread::recvBeginCache()
{ beginMakingCache(); }

void BaseSolverThread::beginMakingCache()
{}

void BaseSolverThread::endMakingCache()
{ emit doneCache(); }

void BaseSolverThread::processMakingCache()
{}

bool BaseSolverThread::isMakingCache() const
{ return false; }

const int& BaseSolverThread::numTicks() const
{ return m_numTicks; }

}
