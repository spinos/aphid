#include "CudaDbgLog.h"
#include "CUDABuffer.h"
#include "AllMath.h"
#include <boost/format.hpp>

std::map<std::string, bool> CudaDbgLog::VisitedPtr;

CudaDbgLog::CudaDbgLog(const std::string & fileName) :
BaseLog(fileName)
{
    m_hostBuf = new BaseBuffer;
}

CudaDbgLog::~CudaDbgLog()
{
    delete m_hostBuf;
}

void CudaDbgLog::writeFlt(BaseBuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
	if(!checkFrequency(freq, notation)) return;
	float * m = (float *)buf->data();
	newLine();
    write(notation);
    unsigned i = 0;
    for(; i < n; i++) {
        write(i);
        _write<float>(m[i]);
		newLine();
    }
}
	
void CudaDbgLog::writeFlt(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
	if(!checkFrequency(freq, notation)) return;
	
    m_hostBuf->create(buf->bufferSize());
    buf->deviceToHost(m_hostBuf->data());
	
	writeFlt(m_hostBuf, n, notation, FIgnore);
}

void CudaDbgLog::writeUInt(BaseBuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
	if(!checkFrequency(freq, notation)) return;
	unsigned * m = (unsigned *)buf->data();
	newLine();
    write(notation);
    unsigned i = 0;
    for(; i < n; i++) {
        write(i);
        _write<unsigned>(m[i]);
		newLine();
    }
}
	
void CudaDbgLog::writeUInt(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
	if(!checkFrequency(freq, notation)) return;
	
    m_hostBuf->create(buf->bufferSize());
    buf->deviceToHost(m_hostBuf->data());
	
	writeUInt(m_hostBuf, n, notation, FIgnore);
}

void CudaDbgLog::writeVec3(BaseBuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
	if(!checkFrequency(freq, notation)) return;
	Vector3F * m = (Vector3F *)buf->data();
	newLine();
    write(notation);
    unsigned i = 0;
    for(; i < n; i++) {
        write(i);
        write(m[i].str());
		newLine();
    }
}
	
void CudaDbgLog::writeVec3(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
	if(!checkFrequency(freq, notation)) return;
	
    m_hostBuf->create(buf->bufferSize());
    buf->deviceToHost(m_hostBuf->data());
	
	writeVec3(m_hostBuf, n, notation, FIgnore);
}

void CudaDbgLog::writeMat33(BaseBuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
	if(!checkFrequency(freq, notation)) return;
	
    Matrix33F * m = (Matrix33F *)buf->data();
	newLine();
    write(notation);
    unsigned i = 0;
    for(; i < n; i++) {
        write(i);
        write(m[i].str());
    }
}
	
void CudaDbgLog::writeMat33(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
    if(!checkFrequency(freq, notation)) return;
	
    m_hostBuf->create(buf->bufferSize());
    buf->deviceToHost(m_hostBuf->data());
	
	writeMat33(m_hostBuf, n, notation, FIgnore);
}

void CudaDbgLog::writeHash(BaseBuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
	if(!checkFrequency(freq, notation)) return;
	
    unsigned * m = (unsigned *)buf->data();
	newLine();
    write(notation);
    unsigned i = 0;
    for(; i < n; i++) {
        write(i);
        write(boost::str(boost::format("(%1%,%2%)\n") % m[i*2] %  m[i*2+1]));
    }
}
	
void CudaDbgLog::writeHash(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
	if(!checkFrequency(freq, notation)) return;
	
    m_hostBuf->create(buf->bufferSize());
    buf->deviceToHost(m_hostBuf->data());
	
	writeHash(m_hostBuf, n, notation, FIgnore);
}

bool CudaDbgLog::checkFrequency(Frequency freq, const std::string & notation)
{
	if(freq == FIgnore) return true;
	if(freq == FOnce) {
        if(VisitedPtr.find(notation) != VisitedPtr.end())
            return false;
		else
			VisitedPtr[notation] = true;
    }
	return true;
}
