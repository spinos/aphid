#ifndef SOLVERTHREAD_H
#define SOLVERTHREAD_H
#include <BaseSolverThread.h>
class CudaTetrahedronSystem;
class CudaNarrowphase;
class SimpleContactSolver;
class BaseBuffer;
class CUDABuffer;

class ContactThread : public BaseSolverThread
{
public:
    ContactThread(QObject *parent = 0);
    virtual ~ContactThread();
    
    void initOnDevice();
    
    CudaTetrahedronSystem * tetra();
    CudaNarrowphase * narrowphase();
    BaseBuffer * hostPairs();
    
protected:
    virtual void stepPhysics(float dt);

private:
    BaseBuffer * m_hostPairs;
	CUDABuffer * m_devicePairs;
    CudaTetrahedronSystem * m_tetra;
	CudaNarrowphase * m_narrowphase;
	SimpleContactSolver * m_contactSolver;
};

#endif        //  #ifndef SOLVERTHREAD_H

