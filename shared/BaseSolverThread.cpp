#include <QtCore>
#include "BaseSolverThread.h"

float BaseSolverThread::TimeStep = 1.f / 60.f;

BaseSolverThread::BaseSolverThread(QObject *parent)
    : QThread(parent)
{
    restart = false;
    abort = false;
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
        //qDebug()<<"wait";
        condition.wakeOne();
    }
}

void BaseSolverThread::run()
{
   forever {
        // qDebug()<<"run";
	
	//for(int i=0; i< 2; i++) {
	    if(restart) {
	        // qDebug()<<"restart ";
            // break;
	    }
	        
	    if (abort) {
            // destroySolverData();
            qDebug()<<"abort";
            return;
        }
        
	    stepPhysics(TimeStep);
	//}
 
		//if (!restart) {
		    // qDebug()<<"end";
            
		    emit doneStep();
		//}

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
