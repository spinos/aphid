#include "CudaDbgLog.h"
#include "AllMath.h"
#include <boost/format.hpp>

namespace aphid {

CudaDbgLog::CudaDbgLog(const std::string & fileName) :
BaseLog(fileName)
{
    m_hostBuf = new BaseBuffer;
}

CudaDbgLog::~CudaDbgLog()
{
    delete m_hostBuf;
}
	
void CudaDbgLog::writeFlt(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
	if(!checkFrequency(freq, notation)) return;
	
    m_hostBuf->create(buf->bufferSize());
    buf->deviceToHost(m_hostBuf->data());
	
    BaseLog::writeFlt(m_hostBuf, n, notation, FIgnore);
}
	
void CudaDbgLog::writeUInt(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
	writeSingle<unsigned int>(buf, n, notation, freq);
}
	
void CudaDbgLog::writeInt(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
	writeSingle<int>(buf, n, notation, freq);
}
	
void CudaDbgLog::writeVec3(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
	if(!checkFrequency(freq, notation)) return;
	
    m_hostBuf->create(buf->bufferSize());
    buf->deviceToHost(m_hostBuf->data());
	
	BaseLog::writeVec3(m_hostBuf, n, notation, FIgnore);
}
	
void CudaDbgLog::writeMat33(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
    if(!checkFrequency(freq, notation)) return;
	
    m_hostBuf->create(buf->bufferSize());
    buf->deviceToHost(m_hostBuf->data());
	
	BaseLog::writeMat33(m_hostBuf, n, notation, FIgnore);
}
	
void CudaDbgLog::writeHash(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
	if(!checkFrequency(freq, notation)) return;
	
    m_hostBuf->create(buf->bufferSize());
    buf->deviceToHost(m_hostBuf->data());
	
	BaseLog::writeHash(m_hostBuf, n, notation, FIgnore);
}
	
void CudaDbgLog::writeMortonHash(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
    if(!checkFrequency(freq, notation)) return;
	
    m_hostBuf->create(buf->bufferSize());
    buf->deviceToHost(m_hostBuf->data());
	
	BaseLog::writeMortonHash(m_hostBuf, n, notation, FIgnore);
}

void CudaDbgLog::writeInt2(CUDABuffer * buf, unsigned n, 
                const std::string & notation,
                Frequency freq)
{
    if(!checkFrequency(freq, notation)) return;
	
    m_hostBuf->create(buf->bufferSize());
    buf->deviceToHost(m_hostBuf->data());
	
	BaseLog::writeInt2(m_hostBuf, n, notation, FIgnore);
}
	
void CudaDbgLog::writeAabb(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
    if(!checkFrequency(freq, notation)) return;
	
    m_hostBuf->create(buf->bufferSize());
    buf->deviceToHost(m_hostBuf->data());
	
	BaseLog::writeAabb(m_hostBuf, n, notation, FIgnore);
}
	
void CudaDbgLog::writeStruct(CUDABuffer * buf, unsigned n, 
                const std::string & notation,
                const std::vector<std::pair<int, int> > & desc,
                unsigned size,
                Frequency freq)
{
    if(!checkFrequency(freq, notation)) return;
	
    m_hostBuf->create(buf->bufferSize());
    buf->deviceToHost(m_hostBuf->data());
	
	BaseLog::writeStruct(m_hostBuf, n, notation, desc, size, FIgnore);
}

}
//:~
