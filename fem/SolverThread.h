#ifndef SOLVERTHREAD_H
#define SOLVERTHREAD_H
#include <BaseSolverThread.h>
#include <ConjugateGradientSolver.h>
#include <FEMTetrahedronMesh.h>

class SolverThread : public BaseSolverThread, ConjugateGradientSolver
{
public:
    SolverThread(QObject *parent = 0);
    virtual ~SolverThread();
    
    FEMTetrahedronMesh * mesh();

protected:
    virtual void stepPhysics(float dt);

private:
    FEMTetrahedronMesh * m_mesh;
	Vector3F * m_F;
	Vector3F * m_F0;
	Vector3F * m_V;

	MatrixMap * m_K_row;
	
    void calculateK();
    void clearStiffnessAssembly();
    void initializePlastic();
    
    void computeForces();
    void updateOrientation();
    void resetOrientation();
    void stiffnessAssembly();
    void addPlasticityForce(float dt);
	void dynamicsAssembly(float dt);
	void updatePosition(float dt);
	void groundCollision();
	void updateF0();
	void updateB(float dt);
};

#endif        //  #ifndef SOLVERTHREAD_H

