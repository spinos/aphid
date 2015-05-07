#ifndef CUDADBGLOG_H
#define CUDADBGLOG_H
#include <map>
#include "BaseLog.h"
class BaseBuffer;
class CUDABuffer;
class CudaDbgLog : public BaseLog {
public:
    enum Frequency {
		FIgnore = 0,
        FOnce = 1,
        FAlways = 2
    };
    
	CudaDbgLog(const std::string & fileName);
	virtual ~CudaDbgLog();
	
	void writeFlt(BaseBuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
	
	void writeFlt(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
					
	void writeUInt(BaseBuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
	
	void writeUInt(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
	
	void writeVec3(BaseBuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
	
	void writeVec3(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
	
	void writeMat33(BaseBuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
	
	void writeMat33(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
					
	void writeHash(BaseBuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
	
	void writeHash(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
	
	void writeMortonHash(BaseBuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
	
	void writeMortonHash(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
    
    void writeInt2(BaseBuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
	
	void writeInt2(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
protected:
    static std::map<std::string, bool> VisitedPtr;
	bool checkFrequency(Frequency freq, const std::string & notation);
private:
	BaseBuffer * m_hostBuf;
};
#endif        //  #ifndef CUDADBGLOG_H

