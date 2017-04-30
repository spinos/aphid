#ifndef AWorldThread_H
#define AWorldThread_H

#include <QMutex>
#include <QThread>
#include <QWaitCondition>
class AWorld;
class AWorldThread : public QThread
{
    Q_OBJECT

public:
    AWorldThread(AWorld * world, QObject *parent = 0);
    virtual ~AWorldThread();
    
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
    
    AWorld * m_world;
    bool abort;
	bool restart;
	unsigned m_numLoops;

public slots:
    void simulate();
};
#endif        //  #ifndef AWorldThread_H

