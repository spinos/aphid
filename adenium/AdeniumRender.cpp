#include "AdeniumRender.h"
#include <CUDABuffer.h>
#include <BaseBuffer.h>
#include <CUDABuffer.h>
#include <CudaPixelBuffer.h>
#include <iostream>
#include "AdeniumRenderInterface.h"
#include <PerspectiveCamera.h>
#include <BvhTriangleSystem.h>
AdeniumRender::AdeniumRender() :
m_imageWidth(0), m_imageHeight(0),
m_initd(0)
{
    m_hostRgbz = new BaseBuffer;
    m_deviceRgbz = new CUDABuffer;
    m_deviceRgbzPix = new CudaPixelBuffer;
}

AdeniumRender::~AdeniumRender() 
{
    delete m_hostRgbz;
    delete m_deviceRgbz;
    delete m_deviceRgbzPix;
}

bool AdeniumRender::resize(int w, int h)
{ 
    if(!isSizeValid(w, h)) return false;
	if(w==m_imageWidth && h == m_imageHeight) return false;
    m_imageWidth = w;
    m_imageHeight = h;
    m_hostRgbz->create(m_imageWidth * m_imageHeight * 4 * 4);
    if(m_initd) m_deviceRgbz->create(m_imageWidth * m_imageHeight * 4 * 4);
    std::cout<<" resize render area: "<<w<<" x "<<h<<"\n";
	return true;
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
    m_deviceRgbzPix->create(numPixels() * 16);
    adetrace::resetImage((float4 *)rgbz(), (uint)numPixels());
}

void * AdeniumRender::rgbz()
{ return m_deviceRgbz->bufferOnDevice(); }

void AdeniumRender::setModelViewMatrix(float * src)
{
	adetrace::setModelViewMatrix(src, 64);
}

void AdeniumRender::renderOrhographic(BaseCamera * camera, BvhTriangleSystem * tri)
{
	void * internalNodeChildIndex = tri->internalNodeChildIndices();
	void * internalNodeAabbs = tri->internalNodeAabbs();
	void * indirection = tri->primitiveHash();
    
	adetrace::renderImageOrthographic((float4 *) rgbz(),
                imageWidth(),
                imageHeight(),
                camera->fieldOfView(),
                camera->aspectRatio(),
				(int2 *)internalNodeChildIndex,
				(Aabb *)internalNodeAabbs);
}

void AdeniumRender::renderPerspective(BaseCamera * camera)
{
	
}

void AdeniumRender::sendToHost()
{
	m_deviceRgbz->deviceToHost(m_hostRgbz->data(), numPixels() * 16);
}

const int AdeniumRender::imageWidth() const
{ return m_imageWidth; }

const int AdeniumRender::imageHeight() const
{ return m_imageHeight; }

void * AdeniumRender::hostRgbz()
{ return m_hostRgbz->data(); }

const bool AdeniumRender::isInitd() const
{ return m_initd==1; }
//:~
