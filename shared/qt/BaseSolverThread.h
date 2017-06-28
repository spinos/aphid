#ifndef APH_BASE_SOLVER_THREAD_H
#define APH_BASE_SOLVER_THREAD_H

#include <QMutex>
#include <QThread>
#include <QWaitCondition>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

namespace aphid {

class BaseSolverThread : public QThread
{
    Q_OBJECT

public:
    BaseSolverThread(QObject *parent = 0);
    virtual ~BaseSolverThread();
    
    static float TimeStep;
    static int NumSubsteps;
    const unsigned numLoops() const;
    
signals:
    void doneStep();

protected:
    void run();
    virtual void stepPhysics(float dt);
    
private:
    QMutex mutex;
    QWaitCondition condition;
    
    bool abort;
	bool restart;
	unsigned m_numLoops;

public slots:
    void simulate();
	
};

}
#endif        //  #ifndef BASESOLVERTHREAD_H

