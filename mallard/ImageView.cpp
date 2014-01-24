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
#include <boost/thread.hpp>

ImageView::ImageView(QWidget *parent)
    : QWidget(parent), BaseServer(7879)
{
	std::cout<<" Renderview ";
    qRegisterMetaType<QImage>("QImage");

	setMinimumSize(400, 300);
    m_image = new QImage(400, 300, QImage::Format_RGB888);
	BaseServer::start();
	
	m_colors = 0;
	m_status = Idle;
}

void ImageView::paintEvent(QPaintEvent * /* event */)
{	
    QPainter painter(this);
	painter.fillRect(rect(), Qt::black);
	painter.drawImage(QPoint(), *m_image);
	
	painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
	painter.setBrush(Qt::NoBrush);
	painter.setPen(Qt::darkCyan);
	
	showImageName(painter);
	showImageSize(painter);
	showRenderer(painter);
	showStatus(painter);
	showRenderTime(painter);
	showBucket(painter);
}

void ImageView::resizeEvent(QResizeEvent *)
{
}

void ImageView::processRead(const char * data, size_t length)
{	
	if(length != 1024 && length != 16 && length != 32) {
		std::clog<<"unknown data size "<<length<<"\n";
		return;
	}

	if(length == 16) beginBucket(data);
	else if(length == 32) endBucket();
	else processPackage(data);
}

void ImageView::beginBucket(const char * data)
{
	if(m_colors) delete[] m_colors;
	m_colors = 0;
	m_packageStart = 0;
	numPix = 0;
	int * box = (int *)data;
	
	bucketRect[0] = box[0];
	bucketRect[1] = box[1];
	bucketRect[2] = box[2];
	bucketRect[3] = box[3];
	
	//std::cout<<"receive bucket("<<bucketRect[0]<<","<<bucketRect[1]<<","<<bucketRect[2]<<","<<bucketRect[3]<<")\n";
	
	if(bucketRect[1] >= m_image->width() || bucketRect[3] >= m_image->height()) {
		std::cout<<" invalid bucket coordinate\n";
		return;
	}
	
	numPix = (bucketRect[1] - bucketRect[0] + 1) * (bucketRect[3] - bucketRect[2] + 1);
	m_colors = new float[numPix * 4];
}

void ImageView::processPackage(const char * data)
{
	if(!m_colors) return;
	doProcessPackage(data);
}

void ImageView::doProcessPackage(const char * data)
{
    float * cdata = (float *)data;
	float * dst = &m_colors[m_packageStart];
	for(int i = 0; i < 256; i++) {
	    if((m_packageStart + i)/4 == numPix) {
	        m_packageStart += i;
	        return;
	    }
		dst[i] = cdata[i];
	}
	m_packageStart += 256;
}

void ImageView::endBucket()
{
    if(!m_colors) return;
    boost::thread t(boost::bind(&ImageView::doEndBucket, this));
    t.join();
}

void ImageView::doEndBucket()
{
	//std::cout<<"fill bucket("<<bucketRect[0]<<","<<bucketRect[1]<<","<<bucketRect[2]<<","<<bucketRect[3]<<")\n";
	if(m_packageStart/4 < numPix) std::clog<<"ERROR: buck not closed"<<m_packageStart/4<<" should be "<<numPix<<"\n";
	uchar r, g, b, a;
	float *pixels = m_colors;
	float gray;
	for (int y = bucketRect[2]; y <= bucketRect[3]; ++y) {
		uchar *scanLine = reinterpret_cast<uchar *>(m_image->scanLine(y));
		scanLine += bucketRect[0] * 3;
		for (int x = bucketRect[0]; x <= bucketRect[1]; ++x) {
		    gray = *pixels;
		    
		    if(gray > 1.f) gray = 1.f;
			r = gray * 255;
			pixels++;
			
			gray = *pixels;
		    if(gray > 1.f) gray = 1.f;
			g = gray * 255;
			pixels++;
			
			gray = *pixels;
		    if(gray > 1.f) gray = 1.f;
			b = gray * 255;
			pixels++;
			
			gray = *pixels;
		    if(gray > 1.f) gray = 1.f;
			a = gray * 255;
			pixels++;
			
			*scanLine = r;
			scanLine++;
			*scanLine = g;
			scanLine++;
			*scanLine = b;
			scanLine++;
		}
	}
	m_numFinishedBucket++;
	if(m_numFinishedBucket >= m_numBucket) m_status = Finished;
	update();
}

void ImageView::resizeImage(QSize s)
{
    if(m_image) delete m_image;
 	m_image = new QImage(s, QImage::Format_RGB888);
 	m_image->fill(0);
	update();
}

void ImageView::setRendererName(QString name)
{
	m_rendererName = name;
}

void ImageView::startRender(QString name)
{
	m_imageName = name;
	m_status = Active;
	int nx = m_image->size().width() / 64;
	if(m_image->size().width() % 64 > 0) nx++;
	int ny = m_image->size().height() / 64;
	if(m_image->size().height() % 64 > 0) ny++;
	
	m_numBucket = nx * ny;
	m_numFinishedBucket = 0;
	
	m_renderBeginTime = boost::posix_time::ptime(boost::posix_time::second_clock::local_time());
	update();
}

void ImageView::showImageName(QPainter & painter)
{
	if(m_status == Idle) return;
	QString text = QString("image: %1").arg(m_imageName); 
	QFontMetrics metrics = painter.fontMetrics();
	painter.drawText(rect().width() / 2 - metrics.width(text) / 2, 4 + metrics.ascent(), text);
}

void ImageView::showImageSize(QPainter & painter)
{
	QFontMetrics metrics = painter.fontMetrics();
	painter.drawText(8, rect().height() - metrics.ascent(),
                         QString("size: %1x%2").arg(m_image->size().width()).arg(m_image->size().height()));
}

void ImageView::showRenderer(QPainter & painter)
{
	QString text = QString("renderer: %1").arg(m_rendererName); 
	QFontMetrics metrics = painter.fontMetrics();
	painter.drawText(rect().width() / 2 - metrics.width(text) / 2, rect().height() - metrics.ascent(), text);
}

void ImageView::showStatus(QPainter & painter)
{
	QString text = QString("status: %1").arg(statusString()); 
	QFontMetrics metrics = painter.fontMetrics();
	painter.drawText(rect().width() - 8 - metrics.width(text), rect().height() - metrics.ascent(), text);
}

QString ImageView::statusString() const
{
	if(m_status == Idle) return tr("idle");
	if(m_status == Active) {
		int percent = (float)m_numFinishedBucket/(float)m_numBucket * 100.f;
		return QString("%1%").arg(percent); 
	}
	return tr("finished");
}

void ImageView::showRenderTime(QPainter & painter)
{
	if(m_status == Idle) return;
	QString text = QString("render time: %1").arg(renderTimeString()); 
	QFontMetrics metrics = painter.fontMetrics();
	painter.drawText(rect().width() - 8 - metrics.width(text), 4 + metrics.ascent(), text);
}

QString ImageView::renderTimeString()
{
	boost::posix_time::time_duration td = boost::posix_time::seconds(renderTimeInt());
	
	int hh = td.hours();
	int mm = td.minutes();
	int ss = td.seconds();
	
	if(hh > 0) return QString("%1 h %2 m %3 s").arg(hh).arg(mm).arg(ss);
	if(mm > 0) return QString("%1 m %2 s").arg(mm).arg(ss);
	return QString("%1 s").arg(ss);
}

int ImageView::renderTimeInt()
{
	if(m_status == Idle) return 0;
	if(m_status == Finished) return m_renderTimeSecs;
	boost::posix_time::time_duration td = boost::posix_time::ptime(boost::posix_time::second_clock::local_time()) - m_renderBeginTime;
	m_renderTimeSecs = td.total_seconds();
	return m_renderTimeSecs;
}

void ImageView::showBucket(QPainter & painter)
{
	if(m_status == Idle || m_status == Finished) return;
	painter.drawRect(bucketRect[0], bucketRect[2], bucketRect[1] - bucketRect[0], bucketRect[3] - bucketRect[2]);
}
