#ifndef CUDACONJUGATEGRADIENTSOLVER_H
#define CUDACONJUGATEGRADIENTSOLVER_H

class CudaCSRMatrix;
class CUDABuffer;
class CudaReduction;
class CudaConjugateGradientSolver {
public:
    CudaConjugateGradientSolver();
    virtual ~CudaConjugateGradientSolver();
    
    void setDimension(unsigned n);
    void initOnDevice();
    
    static int MaxNIterations;
protected:
    void * rightHandSide();
    void solve(void * X, CudaCSRMatrix * A,
                void * fixed, float * error = 0);
private:
    void * residual();
    void * previous();
    void * updated();
    void * diff();
    void * diff2();
private:
    CUDABuffer * m_residual;
	CUDABuffer * m_update;
	CUDABuffer * m_prev;
	CUDABuffer * m_d;
	CUDABuffer * m_d2;
	CUDABuffer * m_rhs;
	CudaReduction * m_reduce;
	unsigned m_dimension;
};
#endif        //  #ifndef CUDACONJUGATEGRADIENTSOLVER_H

