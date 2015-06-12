#include "CudaPixelBuffer.h"
#include <cuda_gl_interop.h>
#include <CudaBase.h>
#include <iostream>
CudaPixelBuffer::CudaPixelBuffer() : m_pbo(0), m_bufferSize(0) {}
CudaPixelBuffer::~CudaPixelBuffer() 
{
    cleanup();
}

void CudaPixelBuffer::cleanup()
{
    if(m_pbo) {
        cudaGraphicsUnregisterResource(m_resource);
        glDeleteBuffers(1, &m_pbo);
        m_pbo = 0;
    }
}

void CudaPixelBuffer::create(unsigned size)
{
    if(m_bufferSize >= size) return;
    cleanup();
	unsigned roundedSize = size;
	if(roundedSize & 1023) roundedSize += 1023;
	roundedSize = roundedSize>>10;
	if(roundedSize < 4) roundedSize = 4;
	roundedSize = roundedSize<<10;
    glGenBuffers(1, &m_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, roundedSize, 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	cudaError_t err = cudaGraphicsGLRegisterBuffer(&m_resource, m_pbo, cudaGraphicsMapFlagsWriteDiscard);
    CudaBase::CheckCudaError(err, " pixel buffer reg resource");
	m_bufferSize = roundedSize;
	std::cout<<" create cu pixel buffer size "<<(roundedSize>>10)<<" K bytes\n";
}

void * CudaPixelBuffer::map()
{
	cudaError_t err = cudaGraphicsMapResources(1, &m_resource, 0);
	CudaBase::CheckCudaError(err, " pixel buffer map resource");
	
	void * entry;
    size_t num_bytes;
    err = cudaGraphicsResourceGetMappedPointer((void **)&entry, &num_bytes,
                                                         m_resource);
														 
	CudaBase::CheckCudaError(err, " pixel buffer map pointer");
	return entry;
}

void CudaPixelBuffer::unmap()
{ 
	cudaError_t err = cudaGraphicsUnmapResources(1, &m_resource, 0);
	CudaBase::CheckCudaError(err, " pixel buffer unmap resource");
}

void CudaPixelBuffer::bind()
{ glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_pbo); }

void CudaPixelBuffer::unbind()
{ glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0); }
