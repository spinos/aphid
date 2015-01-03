#ifndef SOLVERTHREAD_H
#define SOLVERTHREAD_H
#include <pbd_common.h>
#include <QMutex>
#include <QSize>
#include <QThread>
#include <QWaitCondition>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE
class BoxProgram;
class SolverThread : public QThread
{
    Q_OBJECT

public:
    SolverThread(QObject *parent = 0);
    ~SolverThread();
    
    void initProgram();
	
    unsigned numIndices() const;
    Vector3F * pos();
    unsigned * indices();

signals:
    void doneStep();

protected:
    void run();

private:
    QMutex mutex;
    QWaitCondition condition;
    BoxProgram * m_program;
    Vector3F * m_pos;
	Vector3F * m_posLast;
	Vector3F * m_force;
	unsigned * m_indices;
	pbd::Spring * m_spring;
	unsigned m_numSpring;
	
    bool restart;
    bool abort;
	
    void stepPhysics(float dt);
	void setSpring(pbd::Spring * dest, unsigned a, unsigned b, float ks, float kd, int type);
	void computeForces(float dt);
	void integrateVerlet(float dt);
	
	static Vector3F getVerletVelocity(Vector3F x_i, Vector3F xi_last, float dt );
public slots:
    void simulate();
	
};

#endif        //  #ifndef SOLVERTHREAD_H

