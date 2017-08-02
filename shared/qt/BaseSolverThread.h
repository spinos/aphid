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
    const int& numTicks() const;
    
signals:
    void doneStep();
    void doneCache();

protected:
    void run();
    virtual void stepPhysics(float dt);
    virtual void beginMakingCache();
    virtual void endMakingCache();
    virtual void processMakingCache();
    virtual bool isMakingCache() const;
    
private:
    QMutex mutex;
    QWaitCondition condition;
    
    bool abort;
	bool restart;
	unsigned m_numLoops;
	int m_numTicks;

public slots:
    void simulate();
    void recvBeginCache();

};

}
#endif        //  #ifndef BASESOLVERTHREAD_H

