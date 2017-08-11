/*
 *  RenderInterface.h
 *  
 *  access to image, buffer, renderer
 *
 *  Created by jian zhang on 8/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef RENDER_INTERFACE_H
#define RENDER_INTERFACE_H

#include <QImage>

class DisplayImage;
class DeepBuffer;
class BufferBlock;
class Renderer;

class RenderInterface {

	DeepBuffer* m_buffer;
	DisplayImage* m_image;
	
public:
	RenderInterface();
	
	bool imageSizeChanged(int w, int h) const;
	void createImage(int w, int h);
	
	BufferBlock* selectABlock(int nblk);
	DisplayImage* image();
	
	uchar* imageScanline(int i);
	
	const int& xres() const;
	const int& yres() const;
	int bufferNumBlocks() const;
	
	Renderer* getRenderer();
	QImage getQImage() const;
	
};

#endif
