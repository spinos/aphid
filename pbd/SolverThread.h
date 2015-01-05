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
	Vector3F * m_projectedPos;
	Vector3F * m_posLast;
	Vector3F * m_force;
	Vector3F * m_velocity;
	unsigned * m_indices;
	float * m_invMass;
	Vector3F * m_Ri;
	pbd::Spring * m_spring;
	pbd::DistanceConstraint * m_distanceConstraint;
	unsigned m_numBendingConstraint, m_numDistanceConstraint;
	
    bool restart;
    bool abort;
	
    void stepPhysics(float dt);
	void setSpring(pbd::Spring * dest, unsigned a, unsigned b, float ks, float kd, int type);
	void setDistanceConstraint(pbd::DistanceConstraint * dest, unsigned a, unsigned b, float k);
	void computeForces();
	void integrateExplicitWithDamping(float dt);
	void integrateVerlet(float dt);
	void updateConstraints(float dt);
	void updateDistanceConstraint(unsigned i);
	void groundCollision();
	void integrate(float deltaTime);
	
	static Vector3F getVerletVelocity(Vector3F x_i, Vector3F xi_last, float dt );
public slots:
    void simulate();
	
};

#endif        //  #ifndef SOLVERTHREAD_H

