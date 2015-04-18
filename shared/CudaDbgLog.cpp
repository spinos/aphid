#include "CudaDbgLog.h"
#include "CUDABuffer.h"
#include "AllMath.h"

std::map<unsigned long long, bool> CudaDbgLog::VisitedPtr;

CudaDbgLog::CudaDbgLog(const std::string & fileName) :
BaseLog(fileName)
{
    m_hostBuf = new BaseBuffer;
}

CudaDbgLog::~CudaDbgLog()
{
    delete m_hostBuf;
}
	
void CudaDbgLog::writeMat33(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
    if(freq == FOnce) {
        if(VisitedPtr.find((unsigned long long)buf) != VisitedPtr.end())
            return;
    }
    
    VisitedPtr[(unsigned long long)buf] = true;
    m_hostBuf->create(buf->bufferSize());
    buf->deviceToHost(m_hostBuf->data());
    Matrix33F * m = (Matrix33F *)m_hostBuf->data();
    write(notation);
    unsigned i = 0;
    for(; i < n; i++) {
        write(i);
        write(m[i].str());
    }
}
