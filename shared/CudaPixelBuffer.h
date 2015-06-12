#include <gl_heads.h>
#include <cuda_runtime_api.h>
class CudaPixelBuffer {
public:
    CudaPixelBuffer();
    virtual ~CudaPixelBuffer();
    
    void create(unsigned size);
    void * map();
    void unmap();
    void bind();
    void unbind();
protected:
  
private:
    void cleanup();
private:
    unsigned m_bufferSize;
    GLuint m_pbo;
    struct cudaGraphicsResource * m_resource;
};
