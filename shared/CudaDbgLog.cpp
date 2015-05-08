#include "CudaDbgLog.h"
#include "BaseBuffer.h"
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
	writeArraySize(n);
    unsigned i = 0;
    for(; i < n; i++) {
        writeArrayIndex(i);
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
	writeArraySize(n);
    unsigned i = 0;
    for(; i < n; i++) {
        writeArrayIndex(i);
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
	writeArraySize(n);
    unsigned i = 0;
    for(; i < n; i++) {
        writeArrayIndex(i);
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
	writeArraySize(n);
    unsigned i = 0;
    for(; i < n; i++) {
        writeArrayIndex(i);
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
	writeArraySize(n);
    unsigned i = 0;
    for(; i < n; i++) {
        writeArrayIndex(i);
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

const char *byte_to_binary(unsigned x)
{
    static char b[33];
    b[32] = '\0';

    for (int z = 0; z < 32; z++) {
        b[31-z] = ((x>>z) & 0x1) ? '1' : '0';
    }

    return b;
}

void CudaDbgLog::writeMortonHash(BaseBuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
    if(!checkFrequency(freq, notation)) return;
	
    unsigned * m = (unsigned *)buf->data();
	newLine();
    write(notation);
	writeArraySize(n);
    unsigned i = 0;
    for(; i < n; i++) {
        writeArrayIndex(i);
        write(boost::str(boost::format("(%1%,%2%)\n") % byte_to_binary(m[i*2]) %  m[i*2+1]));
    }
}
	
void CudaDbgLog::writeMortonHash(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
    if(!checkFrequency(freq, notation)) return;
	
    m_hostBuf->create(buf->bufferSize());
    buf->deviceToHost(m_hostBuf->data());
	
	writeMortonHash(m_hostBuf, n, notation, FIgnore);
}

void CudaDbgLog::writeInt2(BaseBuffer * buf, unsigned n, 
                const std::string & notation,
                Frequency freq)
{
    if(!checkFrequency(freq, notation)) return;
	
    int * m = (int *)buf->data();
	newLine();
    write(notation);
	writeArraySize(n);
    unsigned i = 0;
    for(; i < n; i++) {
        writeArrayIndex(i);
        write(boost::str(boost::format("(%1%,%2%)\n") % m[i*2] % m[i*2+1]));
    }
}

void CudaDbgLog::writeInt2(CUDABuffer * buf, unsigned n, 
                const std::string & notation,
                Frequency freq)
{
    if(!checkFrequency(freq, notation)) return;
	
    m_hostBuf->create(buf->bufferSize());
    buf->deviceToHost(m_hostBuf->data());
	
	writeInt2(m_hostBuf, n, notation, FIgnore);
}

void CudaDbgLog::writeAabb(BaseBuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
    if(!checkFrequency(freq, notation)) return;
	
    float * m = (float *)buf->data();
	newLine();
    write(notation);
	writeArraySize(n);
    unsigned i = 0;
    for(; i < n; i++) {
        writeArrayIndex(i);
        write(boost::str(boost::format("((%1%,%2%,%3%),(%4%,%5%,%6%))\n") % m[i*6] % m[i*6+1] % m[i*6+2] 
            % m[i*6+3] % m[i*6+4] % m[i*6+5]));
    }
}
	
void CudaDbgLog::writeAabb(CUDABuffer * buf, unsigned n, 
	                const std::string & notation,
	                Frequency freq)
{
    if(!checkFrequency(freq, notation)) return;
	
    m_hostBuf->create(buf->bufferSize());
    buf->deviceToHost(m_hostBuf->data());
	
	writeAabb(m_hostBuf, n, notation, FIgnore);
}

void CudaDbgLog::writeStruct(BaseBuffer * buf, unsigned n, 
	                const std::string & notation,
	                const std::vector<std::pair<int, int> > & desc,
	                unsigned size,
	                Frequency freq)
{
    if(!checkFrequency(freq, notation)) return;
	
    char * m = (char *)buf->data();
	newLine();
    write(notation);
	writeArraySize(n);
    unsigned i = 0;
    for(; i < n; i++) {
        writeArrayIndex(i);
        writeStruct1(&m[i*size], desc);
    }
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
	
	writeStruct(m_hostBuf, n, notation, desc, size, FIgnore);
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
//:~
