#ifndef CUDAREDUCTION_H
#define CUDAREDUCTION_H
#include "bvh_common.h"
#include <CUDABuffer.h>
#include <CudaBase.h>
#include "reduce_common.h"
#include <cuReduceMin_implement.h>
#include <cuReduceMax_implement.h>
#include <cuReduceSum_implement.h>
#include <cuReduceMinMax_implement.h>
#include <cuReduceMinMaxBox_implement.h>

namespace aphid {

class CudaReduction {
public:
    CudaReduction();
    virtual ~CudaReduction();
    
    void initOnDevice();
	
	void * resultOnDevice();
    
template <class T>  
    void min(T & result, T * idata, unsigned m)
    {
        uint threads, blocks;
        uint n = m;
        getReduceBlockThread(blocks, threads, n);
        
        void * d_odata = m_obuf->bufferOnDevice();
        cuReduceFindMin<T>((T *)d_odata, idata, m, blocks, threads);
        
        n = blocks;	
        while(n > 1) {
            blocks = threads = 0;
            getReduceBlockThread(blocks, threads, n);
            
            cuReduceFindMin<T>((T *)d_odata, (T *)d_odata, n, blocks, threads);
            
            n = (n + (threads*2-1)) / (threads*2);
        }
    
        cudaMemcpy(&result, d_odata, 4, cudaMemcpyDeviceToHost);
    }

template <class T>  
    void max(T & result, T * idata, unsigned m)
    {
        uint threads, blocks;
        uint n = m;
        getReduceBlockThread(blocks, threads, n);
        
        void * d_odata = m_obuf->bufferOnDevice();
        cuReduceFindMax<T>((T *)d_odata, idata, m, blocks, threads);
        
        n = blocks;	
        while(n > 1) {
            blocks = threads = 0;
            getReduceBlockThread(blocks, threads, n);
            
            cuReduceFindMax<T>((T *)d_odata, (T *)d_odata, n, blocks, threads);
            
            n = (n + (threads*2-1)) / (threads*2);
        }
    
        cudaMemcpy(&result, d_odata, 4, cudaMemcpyDeviceToHost);
    }

template <class T> 	
    void sum(T & result, T * idata, unsigned m)
    {
        uint threads, blocks;
        uint n = m;
        getReduceBlockThread(blocks, threads, n);
        
        // std::cout<<"n "<<n<<" blocks x threads : "<<blocks<<" x "<<threads<<" sharedmem size "<<threads * sizeof(float)<<"\n";
        
        void * d_odata = m_obuf->bufferOnDevice();
        cuReduceFindSum<T>((T *)d_odata, idata, m, blocks, threads);
        
        n = blocks;	
        while(n > 1) {
            blocks = threads = 0;
            getReduceBlockThread(blocks, threads, n);
            
            // std::cout<<"n "<<n<<" blocks x threads : "<<blocks<<" x "<<threads<<" sharedmem size "<<threads * sizeof(float)<<"\n";
        
            cuReduceFindSum<T>((T *)d_odata, (T *)d_odata, n, blocks, threads);
            
            n = (n + (threads*2-1)) / (threads*2);
        }
    
        cudaMemcpy(&result, d_odata, 4, cudaMemcpyDeviceToHost);
    }
	
template<class T, class T1>
    void minMax(T1 * result, T1 * idata, unsigned m)
    {
        uint threads, blocks;
        uint n = m;
        getReduceBlockThread(blocks, threads, n);
        
        void * d_odata = m_obuf->bufferOnDevice();
        cuReduceFindMinMax<T, T1>((T *)d_odata, idata, m, blocks, threads);
        
        n = blocks;	
        while(n > 1) {
            blocks = threads = 0;
            getReduceBlockThread(blocks, threads, n);
            
            cuReduceFindMinMax<T>((T *)d_odata, (T *)d_odata, n, blocks, threads);
            
            n = (n + (threads*2-1)) / (threads*2);
        }
    
        cudaMemcpy(result, d_odata, sizeof(T1) * 2, cudaMemcpyDeviceToHost);
    }
	
template<class T>
    void minMaxBox(T * result, T * idata, unsigned m)
    {
        uint threads, blocks;
        uint n = m;
        getReduceBlockThread(blocks, threads, n);
        
        void * d_odata = m_obuf->bufferOnDevice();
        cuReduceFindMinMaxBox<T>((T *)d_odata, idata, m, blocks, threads);
        // CudaBase::CheckCudaError("reduce box");
        n = blocks;	
        while(n > 1) {
            blocks = threads = 0;
            getReduceBlockThread(blocks, threads, n);
            
            cuReduceFindMinMaxBox<T>((T *)d_odata, (T *)d_odata, n, blocks, threads);
            // CudaBase::CheckCudaError("reduce box");
            n = (n + (threads*2-1)) / (threads*2);
        }
    
        cudaMemcpy(result, d_odata, sizeof(T), cudaMemcpyDeviceToHost);
    }
    
template<class T, class T1>
    void minMaxBox(T * result, T1 * idata, unsigned m)
    {
        uint threads, blocks;
        uint n = m;
        getReduceBlockThread(blocks, threads, n);
        
        void * d_odata = m_obuf->bufferOnDevice();
        cuReduceFindMinMaxBox<T, T1>((T *)d_odata, idata, m, blocks, threads);
        
        n = blocks;	
        while(n > 1) {
            blocks = threads = 0;
            getReduceBlockThread(blocks, threads, n);
            
            cuReduceFindMinMaxBox<T>((T *)d_odata, (T *)d_odata, n, blocks, threads);
            
            n = (n + (threads*2-1)) / (threads*2);
        }
    
        cudaMemcpy(result, d_odata, sizeof(T), cudaMemcpyDeviceToHost);
    }
	
protected:

private:
    CUDABuffer * m_obuf;
};

}
#endif        //  #ifndef CUDAREDUCTION_H

