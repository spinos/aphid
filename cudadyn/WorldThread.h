#ifndef WORLDTHREAD_H
#define WORLDTHREAD_H

#include <QMutex>
#include <QThread>
#include <QWaitCondition>
#include "CudaDynamicWorld.h"
class WorldThread : public QThread
{
    Q_OBJECT

public:
    WorldThread(CudaDynamicWorld * world, QObject *parent = 0);
    virtual ~WorldThread();
    
    static float TimeStep;
    static int NumSubsteps;
    const unsigned numLoops() const;
    
signals:
    void doneStep();

protected:
    void run();
    
private:
    QMutex mutex;
    QWaitCondition condition;
    
    CudaDynamicWorld * m_world;
    bool abort;
	bool restart;
	unsigned m_numLoops;

public slots:
    void simulate();
};
#endif        //  #ifndef WORLDTHREAD_H

