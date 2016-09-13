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

Plot2DWidget::Plot2DWidget(QWidget *parent) : aphid::BaseImageWidget(parent)
{
	m_margin.set(48, 48);
	setBound(-1.f, 1.f, 4, -1.f, 1.f, 4);
}

Plot2DWidget::~Plot2DWidget()
{}

void Plot2DWidget::clientDraw(QPainter * pr)
{
	QPen pn(Qt::black);
	pn.setWidth(1);
	pr->setPen(pn);
	drawCoordsys(pr);
	drawHorizontalInterval(pr);
	drawVerticalInterval(pr);
	std::vector<UniformPlot1D *>::const_iterator it = m_curves.begin();
	for(;it!=m_curves.end();++it) {
		drawPlot(*it, pr);
	}
}

void Plot2DWidget::drawCoordsys(QPainter * pr) const
{
	QPoint lu = luCorner();
	QPoint rb = rbCorner();
	const QPoint ori(lu.x(), rb.y() );
	
	pr->drawLine(ori, lu );
	pr->drawLine(ori, rb );
}

void Plot2DWidget::drawHorizontalInterval(QPainter * pr) const
{
	QPoint lu = luCorner();
	QPoint rb = rbCorner();
	int l = rb.x() - lu.x();
	int n = m_niterval.x;
	int g = (float)l/(float)n+0.5;
	if(g < 20) {
		g = l;
		n = 1;
	}
	
	const QPoint ori(lu.x(), rb.y() );
	
	float h = (m_hBound.y - m_hBound.x) / n;
	for(int i=0; i<= n; ++i) {
		
		float fv = m_hBound.x + h * i;
		std::string sv = boost::str(boost::format("%=6d") % fv );
		
		QPoint p(ori.x() + g * i, ori.y() );
		pr->drawText(QPoint(p.x() - 16, p.y() + 16), tr(sv.c_str() ) );
		pr->drawLine(p, QPoint(p.x(), p.y() - 4) );
	}
}

void Plot2DWidget::drawVerticalInterval(QPainter * pr) const
{
	QPoint lu = luCorner();
	QPoint rb = rbCorner();
	int l = rb.y() - lu.y();
	int n = m_niterval.y;
	int g = (float)l/(float)n+0.5;
	if(g < 20) {
		g = l;
		n = 1;
	}
	
	const QPoint ori(lu.x(), rb.y() );
	
	float h = (m_vBound.y - m_vBound.x) / n;
	for(int i=0; i<= n; ++i) {
		
		float fv = m_vBound.x + h * i;
		std::string sv = boost::str(boost::format("%=6d") % fv );
		
		QPoint p(ori.x(), ori.y() - g * i);
		pr->drawText(QPoint(p.x() - 40, p.y() + 4 ), tr(sv.c_str() ) );
		pr->drawLine(p, QPoint(p.x() + 4, p.y()) );
	}
}

void Plot2DWidget::drawPlot(const UniformPlot1D * plt, QPainter * pr)
{
	const Vector3F & cf = plt->color(); 
	QPen pn(QColor((int)(cf.x*255), (int)(cf.y*255), (int)(cf.z*255) ) );
	pn.setWidth(1);
	pr->setPen(pn);
	
	QPoint lu = luCorner();
	QPoint rb = rbCorner();
	
	const int & n = plt->numY() - 1;
	const float h = 1.f / (float)n;
	
	QPoint p0(lu.x(), RemapF<int>(rb.y(), lu.y(), m_vBound.x, m_vBound.y,
								plt->y()[0] ) );
	QPoint p1;
	
	int i=1;
	for(;i<=n;++i) {
		p1.setX(MixClamp01F<int>(lu.x(), rb.x(), h*i) );
		p1.setY(RemapF<int>(rb.y(), lu.y(), m_vBound.x, m_vBound.y,
								plt->y()[i] ));
		pr->drawLine(p0, p1 );
		p0 = p1;
	}
	
}

void Plot2DWidget::setMargin(const int & h, const int & v)
{ m_margin.set(h, v); }

QPoint Plot2DWidget::luCorner() const
{ return QPoint(m_margin.x, m_margin.y); }

QPoint Plot2DWidget::rbCorner() const
{ return QPoint(portSize().width() - m_margin.x, portSize().height() - m_margin.y); }

void Plot2DWidget::setBound(const float & hlow, const float & hhigh, 
					const int & hnintv,
					const float & vlow, const float & vhigh, 
					const int & vnintv)
{
	m_hBound.set(hlow, hhigh);
	m_vBound.set(vlow, vhigh);
	m_niterval.set(hnintv, vnintv);
	
}

void Plot2DWidget::addPlot(UniformPlot1D * p)
{ m_curves.push_back(p); }

}