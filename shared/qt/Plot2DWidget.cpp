/*
 *  Plot2DWidget.cpp
 *  
 *
 *  Created by jian zhang on 9/9/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "Plot2DWidget.h"
#include <boost/format.hpp>

namespace aphid {

UniformPlot2DImage::UniformPlot2DImage()
{}

UniformPlot2DImage::~UniformPlot2DImage()
{}

void UniformPlot2DImage::updateImage(const bool & negative)
{
	if(m_img.isNull() ) {
		m_img = QImage(numCols(), numRows(), QImage::Format_RGB32);
	}
	
	const int & w = numCols();
	const int & h = numRows();
	int nch = numChannels();
	if(nch>4) {
		nch = 4;
	}
	
	uint *scanLine = reinterpret_cast<uint *>(m_img.bits() );
	int i, j, k;
	unsigned v;
	int col[4];
	col[0] = col[1] = col[2] = 0;
	col[3] = 255;
			
	for(j=0;j<h;++j) {
		for(i=0;i<w;++i) {
		
			for(k=0;k<nch;++k) {
				
				const float * chan = y(k);
				
				col[k] = chan[iuv(i,j)] * 255;
				
				if(negative && k < 3) {
					col[k] += 127;
				}
				
				Clamp0255(col[k]);
			}
			
/// copy r to g, b in case has only one channel
			for(k=nch;k<3;++k) {
				col[k] = col[nch-1];
			}
			
			v = col[3] << 24;
			v |= ( col[0] << 16 );
			v |= ( col[1] << 8 );
			v |= ( col[2] );
			
			scanLine[j*w + i] = v;
		}
	}
	
}

const QImage & UniformPlot2DImage::image() const
{ return m_img; }

const int & UniformPlot2DImage::width() const
{ return numCols(); }

const int & UniformPlot2DImage::height() const
{ return numRows(); }	
	

Plot2DWidget::Plot2DWidget(QWidget *parent) : Plot1DWidget(parent)
{}

Plot2DWidget::~Plot2DWidget()
{
	std::vector<UniformPlot2DImage * >::iterator it = m_images.begin();
	for(;it!=m_images.end();++it)
		delete *it;
		
	m_images.clear();
}

void Plot2DWidget::clientDraw(QPainter * pr)
{
	QPoint pj = luCorner();
	std::vector<UniformPlot2DImage *>::const_iterator it = m_images.begin();
	for(;it!=m_images.end();++it) {
		drawPlot(*it, pj, pr);
		
		pj.rx() += (*it)->width() * scaleOf(*it);
	}
}

void Plot2DWidget::addImage(UniformPlot2DImage * img)
{ m_images.push_back(img); }

void Plot2DWidget::drawPlot(const UniformPlot2DImage * plt, const QPoint & offset,
							QPainter * pr)
{ 
	pr->translate(offset );
	const float sc = scaleOf(plt);
	pr->scale(sc, sc);
	
	pr->drawImage(QPoint(), plt->image() );
	
	pr->scale(1.f/sc, 1.f/sc);
	pr->translate(QPoint(-offset.x(), -offset.y() ) );
}

float Plot2DWidget::scaleOf(const UniformPlot2DImage * plt) const
{
	if(plt->fillMode() == UniformPlot2D::flFixed) {
		return plt->drawScale();
	}
		
	QPoint lu = luCorner();
	QPoint rb = rbCorner();
	int w = plt->width();
	int h = plt->height();
	if(plt->fillMode() == UniformPlot2D::flVertical )
		return ((float)rb.y() - (float)lu.y() ) / (float)h;

/// fill horizontally
	return ((float)rb.x() - (float)lu.x() ) / (float)w;
}

}