#include <QtCore>
#include "BaseSolverThread.h"

float BaseSolverThread::TimeStep = 1.f / 60.f;

BaseSolverThread::BaseSolverThread(QObject *parent)
    : QThread(parent)
{
    abort = false;
	restart = false;
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
        
		stepPhysics(TimeStep);

		emit doneStep();

		mutex.lock();
		
        if (!restart)
			condition.wait(&mutex);
			
		restart = false;
        mutex.unlock();
   }
}

void BaseSolverThread::stepPhysics(float dt)
{
}
