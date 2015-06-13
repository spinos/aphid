#include "AdeniumRender.h"
#include <CUDABuffer.h>
#include <BaseBuffer.h>
#include <CUDABuffer.h>
#include <CudaBase.h>
#include <CudaPixelBuffer.h>
#include <iostream>
#include "AdeniumRenderInterface.h"
#include <PerspectiveCamera.h>
#include <BvhTriangleSystem.h>
AdeniumRender::AdeniumRender() :
m_imageWidth(0), m_imageHeight(0)
{
    m_deviceRgbzPix = new CudaPixelBuffer;
}

AdeniumRender::~AdeniumRender() 
{
    delete m_deviceRgbzPix;
}

bool AdeniumRender::resize(int w, int h)
{ 
    if(!isSizeValid(w, h)) return false;
	if(w==m_imageWidth && h == m_imageHeight) return false;
    m_imageWidth = w;
    m_imageHeight = h;
    std::cout<<" resize render area: "<<w<<" x "<<h<<"\n";
	return true;
}

bool AdeniumRender::isSizeValid(int x, int y) const
{ return (x > 0 && y > 0); }

int AdeniumRender::numPixels() const
{ return m_imageWidth * m_imageHeight; }

void AdeniumRender::reset()
{
	m_deviceRgbzPix->create(numPixels() * 16);
	
	void * pix = m_deviceRgbzPix->map();
	adetrace::resetImage((float4 *)pix, (uint)numPixels());
	CudaBase::CheckCudaError(" reset image");
	m_deviceRgbzPix->unmap();
}

void AdeniumRender::setModelViewMatrix(float * src)
{
	adetrace::setModelViewMatrix(src, 64);
}

void AdeniumRender::renderOrhographic(BaseCamera * camera, BvhTriangleSystem * tri)
{
	void * internalNodeChildIndex = tri->internalNodeChildIndices();
	void * internalNodeAabbs = tri->internalNodeAabbs();
	void * indirection = tri->primitiveHash();
    void * pix = m_deviceRgbzPix->map();
    void * vertices = tri->deviceTretradhedronIndices(); 
    void * points = tri->deviceX();
	adetrace::renderImage((float4 *)pix,
                imageWidth(),
                imageHeight(),
                camera->fieldOfView(),
                camera->aspectRatio(),
				(int2 *)internalNodeChildIndex,
				(Aabb *)internalNodeAabbs,
				(KeyValuePair *)indirection,
				(int4 *)vertices,
				(float3 *)points,
				1);
	CudaBase::CheckCudaError(" render ortho image");
	m_deviceRgbzPix->unmap();
}

void AdeniumRender::renderPerspective(BaseCamera * camera, BvhTriangleSystem * tri)
{
	void * internalNodeChildIndex = tri->internalNodeChildIndices();
	void * internalNodeAabbs = tri->internalNodeAabbs();
	void * indirection = tri->primitiveHash();
    void * pix = m_deviceRgbzPix->map();
    void * vertices = tri->deviceTretradhedronIndices(); 
    void * points = tri->deviceX();
	adetrace::renderImage((float4 *)pix,
                imageWidth(),
                imageHeight(),
                camera->frameWidth(),
                camera->aspectRatio(),
				(int2 *)internalNodeChildIndex,
				(Aabb *)internalNodeAabbs,
				(KeyValuePair *)indirection,
				(int4 *)vertices,
				(float3 *)points,
				0);
	CudaBase::CheckCudaError(" render persp image");
	m_deviceRgbzPix->unmap();
}

const int AdeniumRender::imageWidth() const
{ return m_imageWidth; }

const int AdeniumRender::imageHeight() const
{ return m_imageHeight; }

void AdeniumRender::bindBuffer()
{ m_deviceRgbzPix->bind(); }

void AdeniumRender::unbindBuffer()
{ m_deviceRgbzPix->unbind(); }
//:~