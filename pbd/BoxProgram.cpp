#include "BoxProgram.h"
#include "bbox_implement.h"
#include <CUDABuffer.h>

unsigned nextPow2( unsigned x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

BoxProgram::BoxProgram() {}
BoxProgram::~BoxProgram() {}

void BoxProgram::createCvs(unsigned numCvs)
{
    m_cvs = new CUDABuffer;
    m_cvs->create(numCvs * sizeof(Vector3F));
}

void BoxProgram::createIndices(unsigned numIndices, unsigned * src)
{
    m_indices = new CUDABuffer;
    m_indices->create(numIndices * sizeof(unsigned));
    m_indices->hostToDevice(src, numIndices * sizeof(unsigned));
}

void BoxProgram::createAabbs(unsigned n)
{
    m_aabb = new CUDABuffer;
    m_aabb->create(n * sizeof(Aabb));
    
    std::cout<<"reduce aabb init\n";
    std::cout<<"n aabb: "<<n<<"\n";
    
    unsigned maxThreads = 256;
    unsigned nthreads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    std::cout<<"n threads: "<<nthreads<<"\n";
}

void BoxProgram::getAabbs(Vector3F * dst, unsigned nbox)
{
    m_aabb->deviceToHost(dst, nbox * sizeof(Aabb));
}

void BoxProgram::run(Vector3F * pos, unsigned numTriangle, unsigned numVertices)
{
    m_cvs->hostToDevice(pos, numVertices * sizeof(Vector3F));
    calculateAabbs((Aabb *)m_aabb->bufferOnDevice(), (float3 *)m_cvs->bufferOnDevice(), (unsigned *)m_indices->bufferOnDevice(), numTriangle);
}
