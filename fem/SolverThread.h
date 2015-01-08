#ifndef SOLVERTHREAD_H
#define SOLVERTHREAD_H
#include <BaseSolverThread.h>
#include <FEMTetrahedronMesh.h>

#include <map>
typedef std::map<int, Matrix33F> MatrixMap;

class SolverThread : public BaseSolverThread
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
	Vector3F * m_b;
	Vector3F * m_V;
	Vector3F * m_residual;
	Vector3F * m_update;
	Vector3F * m_prev;
	MatrixMap * m_K_row;
	MatrixMap * m_A_row;
	bool * m_IsFixed;
	
    void calculateK();
    void clearStiffnessAssembly();
    void initializePlastic();
    
    void computeForces();
    void updateOrientation();
    void resetOrientation();
    void stiffnessAssembly();
    void addPlasticityForce(float dt);
	void dynamicsAssembly(float dt);
	void conjugateGradientSolver(float dt);
	void updatePosition(float dt);
	void groundCollision();
};

#endif        //  #ifndef SOLVERTHREAD_H

