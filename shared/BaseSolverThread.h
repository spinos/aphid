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
    
signals:
    void doneStep();

protected:
    void run();
    virtual void stepPhysics(float dt);

private:
    QMutex mutex;
    QWaitCondition condition;
    
    bool restart;
    bool abort;

public slots:
    void simulate();
	
};

#endif        //  #ifndef BASESOLVERTHREAD_H

