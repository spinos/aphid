#ifndef BASESOLVERTHREAD_H
#define BASESOLVERTHREAD_H

#include <QMutex>
#include <QThread>
#include <QWaitCondition>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class BaseSolverThread : public QThread
{
    Q_OBJECT

public:
    BaseSolverThread(QObject *parent = 0);
    virtual ~BaseSolverThread();
    
    static float TimeStep;
    static int NumSubsteps;
    const unsigned numLoops() const;
    virtual void stepPhysics(float dt);
    
signals:
    void doneStep();

protected:
    void run();
    
private:
    QMutex mutex;
    QWaitCondition condition;
    
    bool abort;
	bool restart;
	unsigned m_numLoops;

public slots:
    void simulate();
	
};

#endif        //  #ifndef BASESOLVERTHREAD_H

