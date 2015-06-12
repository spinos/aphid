#include "CudaPixelBuffer.h"
#include <cuda_gl_interop.h>
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
    glGenBuffers(1, &m_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, size, 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    cudaGraphicsGLRegisterBuffer(&m_resource, m_pbo, cudaGraphicsMapFlagsWriteDiscard);
    m_bufferSize = size;
}

void CudaPixelBuffer::map(void * entry)
{
    cudaGraphicsMapResources(1, &m_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&entry, &num_bytes,
                                                         m_resource);
}

void CudaPixelBuffer::unmap()
{ cudaGraphicsUnmapResources(1, &m_resource, 0); }

void CudaPixelBuffer::bind()
{ glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_pbo); }

void CudaPixelBuffer::unbind()
{ glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0); }
