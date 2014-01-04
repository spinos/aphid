/*
 *  ImageView.cpp
 *  mallard
 *
 *  Created by jian zhang on 12/29/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "ImageView.h"
#include <BaseServer.h>
#include <iostream>
ImageView::ImageView(QWidget *parent)
    : QWidget(parent), BaseServer(7879)
{
	std::cout<<" Renderview ";
    qRegisterMetaType<QImage>("QImage");
    //connect(&thread, SIGNAL(renderedImage(QImage)),
      //      this, SLOT(updatePixmap(QImage)));

	setMinimumSize(400, 300);
    m_image = new QImage(400, 300, QImage::Format_RGB32);
	BaseServer::start();
	//QTimer *timer = new QTimer(this);
	//connect(timer, SIGNAL(timeout()), this, SLOT(simulate()));
	//timer->start(40);
	m_colors = 0;
}

void ImageView::paintEvent(QPaintEvent * /* event */)
{
    QPainter painter(this);
	painter.fillRect(rect(), Qt::gray);
    /*
    if (pixmap.isNull()) {
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter,
                         tr("Rendering initial image, please wait..."));
        return;
    }
	*/
	painter.drawImage(QPoint(), *m_image);
}

void ImageView::resizeEvent(QResizeEvent *)
{
}

void ImageView::processRead(const char * data, size_t length)
{	
	if(length != 1024) {
		std::clog<<"unknown data size "<<length<<"\n";
		return;
	}
	
	if(data[16] == '\n') beginBucket(data);
	else if(data[0] == '\n') endBucket();
	else processPackage(data);
}

void ImageView::beginBucket(const char * data)
{
	if(m_colors) delete[] m_colors;
	m_colors = 0;
	m_packageStart = 0;
	
	int * box = (int *)data;
	
	bucketRect[0] = box[0];
	bucketRect[1] = box[1];
	bucketRect[2] = box[2];
	bucketRect[3] = box[3];
	
	std::cout<<"receive bucket("<<bucketRect[0]<<","<<bucketRect[1]<<","<<bucketRect[2]<<","<<bucketRect[3]<<")\n";
	
	if(bucketRect[1] >= m_image->width() || bucketRect[3] >= m_image->height()) {
		std::cout<<" invalid bucket coordinate\n";
		return;
	}
	
	const unsigned npix = (bucketRect[1] - bucketRect[0] + 1) * (bucketRect[3] - bucketRect[2] + 1);
	
	int pixLength = npix;
	if(pixLength % 256) pixLength = (npix / 256 + 1 )* 256;

	m_colors = new float[pixLength * 4];
}

void ImageView::processPackage(const char * data)
{
	if(!m_colors) return;
	float * cdata = (float *)data;
	float * dst = &m_colors[m_packageStart];
	for(int i = 0; i < 256; i++)
		dst[i] = cdata[i];
	m_packageStart += 256;
}

void ImageView::endBucket()
{
	if(!m_colors) return;
	int r, g, b, a;
	float *pixels = m_colors;
	for (int y = bucketRect[2]; y <= bucketRect[3]; ++y) {
		uint *scanLine = reinterpret_cast<uint *>(m_image->scanLine(y));
		scanLine += bucketRect[0];
		for (int x = bucketRect[0]; x <= bucketRect[1]; ++x) {
			r = *pixels * 255;
			pixels++;
			g = *pixels * 255;
			pixels++;
			b = *pixels * 255;
			pixels++;
			a = *pixels * 255;
			pixels++;
			*scanLine++ = qRgb(r, g, b);
		}
	}
	
	update();
}

void ImageView::resizeImage(QSize s)
{
	if(m_image) delete m_image;
 	m_image = new QImage(s, QImage::Format_RGB32);
	update();
}
