#ifndef CUDADBGLOG_H
#define CUDADBGLOG_H
#include "BaseLog.h"
#include "CUDABuffer.h"

namespace aphid {

class CudaDbgLog : public BaseLog {
public:    
	CudaDbgLog(const std::string & fileName);
	virtual ~CudaDbgLog();
	
	void writeFlt(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
					
	void writeUInt(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
	
	void writeInt(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
	
	void writeVec3(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
	
	void writeMat33(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
					
	void writeHash(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
	
	void writeMortonHash(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
    
	void writeInt2(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
	
	void writeAabb(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
	
	void writeStruct(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                const std::vector<std::pair<int, int> > & desc,
	                unsigned size,
	                Frequency freq = FOnce);
protected:
	template<typename T>
	void writeSingle(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce)
    {
        if(!checkFrequency(freq, notation)) return;
        
        m_hostBuf->create(buf->bufferSize());
        buf->deviceToHost(m_hostBuf->data());
        
        BaseLog::writeSingle<T>(m_hostBuf, n, notation, FIgnore);
    }
    
private:
	BaseBuffer * m_hostBuf;
};

}
#endif        //  #ifndef CUDADBGLOG_H

