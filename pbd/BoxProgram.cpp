#include "BoxProgram.h"
#include "box_implement.h"
#include <CUDABuffer.h>

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

void BoxProgram::createAabb(unsigned n)
{
    m_aabb = new CUDABuffer;
    m_aabb->create(n * sizeof(Aabb));
}

void BoxProgram::getAabb(Vector3F * dst, unsigned nbox)
{
    m_aabb->deviceToHost(dst, nbox * sizeof(Aabb));
}

void BoxProgram::run(Vector3F * pos, unsigned numTriangle, unsigned numVertices)
{
    m_cvs->hostToDevice(pos, numVertices * sizeof(Vector3F));
    calculateAabb((Aabb *)m_aabb->bufferOnDevice(), (float3 *)m_cvs->bufferOnDevice(), (unsigned *)m_indices->bufferOnDevice(), numTriangle);
}
