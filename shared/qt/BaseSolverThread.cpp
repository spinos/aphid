#include <QtCore>
#include "BaseSolverThread.h"

namespace aphid {

float BaseSolverThread::TimeStep = 1.f / 30.f;
int BaseSolverThread::NumSubsteps = 4;
BaseSolverThread::BaseSolverThread(QObject *parent)
    : QThread(parent)
{
    abort = false;
	pause = false;
	restart = false;
	restartAtCurrentState = false;
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

void BaseSolverThread::recvRestartAtCurrentState()
{ 
	mutex.lock();
    restartAtCurrentState = true;
    condition.wakeOne();
    mutex.unlock();
}

void BaseSolverThread::recvToggleSimulation()
{
	mutex.lock();
    pause = !pause;
    condition.wakeOne();
    mutex.unlock();
}

void BaseSolverThread::run()
{	 
	forever {
	    if (abort) {
            // destroySolverData();
            qDebug()<<"abort";
            return;
        }
		
		if(pause)
			continue;
		
		if(restartAtCurrentState) {
			restartCurrentState();
			restartAtCurrentState = false;
			
		}
        
		const float dt = TimeStep / (float)NumSubsteps;
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

void BaseSolverThread::restartCurrentState()
{}

}
