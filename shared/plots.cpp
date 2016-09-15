/*
 *  plots.cpp
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "plots.h"

namespace aphid {

UniformPlot1D::UniformPlot1D() :
m_numY(0),
m_lstyle(LsSolid),
m_y(NULL)
{}

UniformPlot1D::~UniformPlot1D()
{
	if(m_numY) delete[] m_y;
}

void UniformPlot1D::create(const int & n)
{
	if(m_numY >= n) {
		m_numY = n;
		return;
	}
	
	if(m_numY) delete[] m_y;
	m_numY = n;
	m_y = new float[n];
}

void UniformPlot1D::create(const float * y, const int & n)
{
	create(n);
	for(int i=0;i<n;++i)
		m_y[i] = y[i];
}

void UniformPlot1D::setColor(float r, float g, float b)
{ m_color.set(r,g,b); }

const Vector3F & UniformPlot1D::color() const
{ return m_color; }

const float * UniformPlot1D::y() const
{ return m_y; }

float * UniformPlot1D::y()
{ return m_y; }

const int & UniformPlot1D::numY() const
{ return m_numY; }

void UniformPlot1D::setLineStyle(UniformPlot1D::LineStyle x)
{ m_lstyle = x; }

const UniformPlot1D::LineStyle & UniformPlot1D::lineStyle() const
{ return m_lstyle; }

UniformPlot2D::UniformPlot2D() :
m_fmd(flFixed),
m_drScale(1.f)
{}

UniformPlot2D::~UniformPlot2D()
{}

void UniformPlot2D::create(const int & m, const int & n, const int & p)
{ m_data.create(m,n,p); }

void UniformPlot2D::setFillMode(FillMode x)
{ m_fmd = x; }

void UniformPlot2D::setDrawScale(float x)
{ m_drScale = x; }

const int & UniformPlot2D::numRows() const
{ return m_data.numRows(); }

const int & UniformPlot2D::numCols() const
{ return m_data.numCols(); }

const int & UniformPlot2D::numChannels() const
{ return m_data.numRanks(); }

UniformPlot2D::FillMode UniformPlot2D::fillMode() const
{ return m_fmd; }

const float & UniformPlot2D::drawScale() const
{ return m_drScale; }

float * UniformPlot2D::y(const int & k)
{ return channel(k)->v(); }

const float * UniformPlot2D::y(const int & k) const
{ return channel(k)->v(); }

int UniformPlot2D::iuv(const int & u, const int & v) const
{ return u * m_data.numRows() + v; }

Array2<float> * UniformPlot2D::channel(const int & k)
{ return m_data.rank(k); }

const Array2<float> * UniformPlot2D::channel(const int & k) const
{ return m_data.rank(k); }

}