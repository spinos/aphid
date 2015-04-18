#ifndef CUDADBGLOG_H
#define CUDADBGLOG_H
#include <map>
#include "BaseLog.h"
class BaseBuffer;
class CUDABuffer;
class CudaDbgLog : public BaseLog {
public:
    enum Frequency {
        FOnce = 0,
        FAlways = 1
    };
    
	CudaDbgLog(const std::string & fileName);
	virtual ~CudaDbgLog();
	
	void writeMat33(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq = FOnce);
protected:
    static std::map<unsigned long long, bool> VisitedPtr;
private:
	BaseBuffer * m_hostBuf;
};
#endif        //  #ifndef CUDADBGLOG_H

