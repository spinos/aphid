/*
 *  Plot1DWidget.cpp
 *  
 *
 *  Created by jian zhang on 9/9/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "Plot1DWidget.h"
#include <boost/format.hpp>

namespace aphid {

Plot1DWidget::Plot1DWidget(QWidget *parent) : BaseImageWidget(parent)
{
	setBound(-1.f, 1.f, 4, -1.f, 1.f, 4);
}

Plot1DWidget::~Plot1DWidget()
{
	std::vector<UniformPlot1D *>::iterator it = m_curves.begin();
	for(;it!=m_curves.end();++it) {
		delete *it;
	}
	m_curves.clear();
	
}

void Plot1DWidget::clientDraw(QPainter * pr)
{
	QPen pn(Qt::black);
	pn.setWidth(1);
	pr->setPen(pn);
	drawCoordsys(pr);
	
	std::vector<UniformPlot1D *>::const_iterator it = m_curves.begin();
	for(;it!=m_curves.end();++it) {
		drawPlot(*it, pr);
	}
}

void Plot1DWidget::drawCoordsys(QPainter * pr) const
{
	QPoint lu = luCorner();
	QPoint rb = rbCorner();
	const QPoint ori(lu.x(), rb.y() );
	
	pr->drawLine(ori, lu );
	pr->drawLine(ori, rb );
	
	drawHorizontalInterval(pr);
	drawVerticalInterval(pr);
	
}

void Plot1DWidget::drawHorizontalInterval(QPainter * pr) const
{
	QPoint lu = luCorner();
	QPoint rb = rbCorner();
	int l = rb.x() - lu.x();
	int n = m_niterval.x;
	int g = (float)l/(float)n;
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
		if(i==n) p.setX(rb.x() );
		
		pr->drawText(QPoint(p.x() - 16, p.y() + 16), tr(sv.c_str() ) );
		pr->drawLine(p, QPoint(p.x(), p.y() - 4) );
	}
}

void Plot1DWidget::drawVerticalInterval(QPainter * pr) const
{
	QPoint lu = luCorner();
	QPoint rb = rbCorner();
	int l = rb.y() - lu.y();
	int n = m_niterval.y;
	int g = (float)l/(float)n;
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
		if(i==n) p.setY(lu.y() );
		
		pr->drawText(QPoint(p.x() - 40, p.y() + 4 ), tr(sv.c_str() ) );
		pr->drawLine(p, QPoint(p.x() + 4, p.y()) );
	}
}

void Plot1DWidget::drawPlot(const UniformPlot1D * plt, QPainter * pr)
{
	switch(plt->geomType() ) {
		case UniformPlot1D::GtLine :
			drawLine(plt, pr);
			break;
		case UniformPlot1D::GtMark :
			drawMark(plt, pr);
			break;
		default:
			break;
	}
}
	
void Plot1DWidget::drawLine(const UniformPlot1D * plt, QPainter * pr)
{
	const Vector3F & cf = plt->color(); 
	QPen pn(QColor((int)(cf.x*255), (int)(cf.y*255), (int)(cf.z*255) ) );
	pn.setWidth(1);
	
	switch (plt->lineStyle() ) {
		case UniformPlot1D::LsDash :
			pn.setStyle(Qt::DashLine);
			break;
		case UniformPlot1D::LsDot :
			pn.setStyle(Qt::DotLine);
			break;
		default:
			break;
	}
	
	pr->setPen(pn);
	
	QPoint lu = luCorner();
	QPoint rb = rbCorner();
	
	const int & n = plt->numY() - 1;

	QPoint p0, p1;
	
/// linspace x
	p0.setX(MixClamp01F<int>(lu.x(), rb.x(), plt->x()[0]) );
	p0.setY(RemapF<int>(rb.y(), lu.y(), m_vBound.x, m_vBound.y,
								plt->y()[0] ));
	
	int i=1;
	for(;i<=n;++i) {
		p1.setX(MixClamp01F<int>(lu.x(), rb.x(), plt->x()[i]) );
		p1.setY(RemapF<int>(rb.y(), lu.y(), m_vBound.x, m_vBound.y,
								plt->y()[i] ));
		pr->drawLine(p0, p1 );
		p0 = p1;
	}
	
}

static const QPointF quadPoints[4] = {
     QPointF(-4.0, -4.0),
     QPointF( 4.0, -4.0),
     QPointF( 4.0, 4.0),
	 QPointF(-4.0, 4.0)
 };

void Plot1DWidget::drawMark(const UniformPlot1D * plt, QPainter * pr)
{
	const Vector3F & cf = plt->color(); 
	QPen pn(QColor((int)(cf.x*255), (int)(cf.y*255), (int)(cf.z*255) ) );
	pn.setWidth(1);
	
	pr->setPen(pn);
	
	QPoint lu = luCorner();
	QPoint rb = rbCorner();
	
	const int & n = plt->numY();

	QPoint p1;
	
	int i=0;
	for(;i<n;++i) {
/// actual x
		p1.setX(RemapF<int>(lu.x(), rb.x(), m_hBound.x, m_hBound.y,
								plt->x()[i] ));
		p1.setY(RemapF<int>(rb.y(), lu.y(), m_vBound.x, m_vBound.y,
								plt->y()[i] ));
								
		pr->translate(p1);
		pr->drawPolygon(quadPoints, 4);
		pr->translate(p1*(-1.0));

	}
	
}

QPoint Plot1DWidget::luCorner() const
{ return QPoint(margin().x, margin().y); }

QPoint Plot1DWidget::rbCorner() const
{ return QPoint(portSize().width() - margin().x, portSize().height() - margin().y); }

void Plot1DWidget::setBound(const float & hlow, const float & hhigh, 
					const int & hnintv,
					const float & vlow, const float & vhigh, 
					const int & vnintv)
{
	m_hBound.set(hlow, hhigh);
	m_vBound.set(vlow, vhigh);
	m_niterval.set(hnintv, vnintv);
	
}

void Plot1DWidget::addVectorPlot(UniformPlot1D * p)
{ m_curves.push_back(p); }

Vector2F Plot1DWidget::toRealSpace(const int & x,
	                    const int & y) const
{
    const QPoint lu = luCorner();
    const QPoint rb = rbCorner();
    Vector2F r;
    r.x = RemapF<float>(m_hBound.x, m_hBound.y, lu.x(), rb.x(), 
								x);
	r.y = RemapF<float>(m_vBound.x, m_vBound.y, rb.y(), lu.y(), 
								y);
	return r;
}

float Plot1DWidget::xToBound(const float & x) const
{
    return m_hBound.x + (m_hBound.y - m_hBound.x) * x;
}

}