#include "AdeniumRender.h"
#include <CUDABuffer.h>
#include <BaseBuffer.h>
#include <CUDABuffer.h>
#include <iostream>
#include "AdeniumRenderInterface.h"
AdeniumRender::AdeniumRender() :
m_imageWidth(0), m_imageHeight(0),
m_initd(0)
{
    m_hostRgbz = new BaseBuffer;
    m_deviceRgbz = new CUDABuffer;
}

AdeniumRender::~AdeniumRender() {}

void AdeniumRender::resize(int w, int h)
{ 
    if(!isSizeValid(w, h)) return;
    m_imageWidth = w;
    m_imageHeight = h;
    m_hostRgbz->create(m_imageWidth * m_imageHeight * 4 * 4);
    if(m_initd) m_deviceRgbz->create(m_imageWidth * m_imageHeight * 4 * 4);
    std::cout<<" resize render area: "<<w<<" x "<<h<<"\n";
}

void AdeniumRender::initOnDevice()
{
    if(!isSizeValid(m_imageWidth, m_imageHeight)) return;
    m_deviceRgbz->create(m_imageWidth * m_imageHeight * 4 * 4);
    m_initd = 1;
}

bool AdeniumRender::isSizeValid(int x, int y) const
{ return (x > 0 && y > 0); }

int AdeniumRender::numPixels() const
{ return m_imageWidth * m_imageHeight; }

void AdeniumRender::reset()
{
    adetrace::resetImage((float4 *) rgbz(), (uint)numPixels());
}

void * AdeniumRender::rgbz()
{ return m_deviceRgbz->bufferOnDevice(); }
//:~
