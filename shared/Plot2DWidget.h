/*
 *  Plot2DWidget.h
 *  
 *
 *  Created by jian zhang on 9/9/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APHID_PLOT_2D_WIDGET_H
#define APHID_PLOT_2D_WIDGET_H

#include <BaseImageWidget.h>
#include <Calculus.h>

namespace aphid {

class UniformPlot2D;

class Plot2DWidget : public BaseImageWidget {

	Q_OBJECT
	
public:
	Plot2DWidget(QWidget *parent = 0);
	virtual ~Plot2DWidget();
	
/// horizontal and vertical margin size
	void setMargin(const int & h, const int & v);
/// horizontal and vertical bound 
/// low high number of intervals
	void setBound(const float & hlow, const float & hhigh, 
					const int & hnintv,
					const float & vlow, const float & vhigh, 
					const int & vnintv);
					
	void addPlot(UniformPlot2D * p);
	
public slots:
   	
protected:
	virtual void clientDraw(QPainter * pr);

private:
	void drawCoordsys(QPainter * pr) const;
	void drawHorizontalInterval(QPainter * pr) const;
	void drawVerticalInterval(QPainter * pr) const;
	void drawPlot(const UniformPlot2D * plt, QPainter * pr);
	
/// left-up
	QPoint luCorner() const;
/// right-bottom
	QPoint rbCorner() const;
	
private:
	Int2 m_margin;
	Vector2F m_hBound;
	Vector2F m_vBound;
	Int2 m_niterval;
	
	std::vector<UniformPlot2D *> m_curves;
	
};

}
#endif
