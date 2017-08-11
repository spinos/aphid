/*
 *  RenderInterface.cpp
 *  
 *
 *  Created by jian zhang on 8/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "RenderInterface.h"
#include "BufferBlock.h"
#include "DeepBuffer.h"
#include "DisplayImage.h"
#include "NoiseRenderer.h"

RenderInterface::RenderInterface()
{
	m_buffer = new DeepBuffer;
	m_image = new DisplayImage;
}

bool RenderInterface::imageSizeChanged(int w, int h) const
{
	return (xres() != BufferBlock::RoundToBlockSize(w) 
	|| yres() != BufferBlock::RoundToBlockSize(h) );
}

void RenderInterface::createImage(int w, int h)
{ 
	m_buffer->create(w, h);
	
	int rxres = m_buffer->width();
	int ryres = m_buffer->height();
	m_image->create(rxres, ryres);
}

DisplayImage* RenderInterface::image()
{ return m_image; }

QImage RenderInterface::getQImage() const
{ return QImage(m_image->bits(), 
	m_image->xres(), m_image->yres(), 
	QImage::Format_RGB32); 
}

uchar* RenderInterface::imageScanline(int i)
{ return m_image->scanline(i); }

const int& RenderInterface::xres() const
{ return m_image->xres(); }
	
const int& RenderInterface::yres() const
{ return m_image->yres(); }

BufferBlock* RenderInterface::selectABlock(int nblk)
{
	int i = rand() % nblk;
	return m_buffer->block(i);
}

int RenderInterface::bufferNumBlocks() const
{ return m_buffer->numBlocks(); }

Renderer* RenderInterface::getRenderer()
{ return new NoiseRenderer; }
