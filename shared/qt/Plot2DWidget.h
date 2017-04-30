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

#include <qt/Plot1DWidget.h>

namespace aphid {

class UniformPlot2DImage : public UniformPlot2D {

public:
	UniformPlot2DImage();
	virtual ~UniformPlot2DImage();
	
/// pack up to 4 channels to RGBA32 images
/// assuming channel ordered in r, g, b, a
/// +0.5 if negative
	void updateImage(const bool & negative = false);
	
	const QImage & image() const;
	const int & width() const;
	const int & height() const;
	
protected:

private:
	QImage m_img;
	
};

class Plot2DWidget : public Plot1DWidget {

	Q_OBJECT
	
public:
	Plot2DWidget(QWidget *parent = 0);
	virtual ~Plot2DWidget();
	
	void addImage(UniformPlot2DImage * img);
	
public slots:
   	
protected:
	virtual void clientDraw(QPainter * pr);

	float scaleOf(const UniformPlot2DImage * plt) const;
	
private:
	void drawPlot(const UniformPlot2DImage * plt, const QPoint & offset, 
					QPainter * pr);
	
private:
	std::vector<UniformPlot2DImage * > m_images;
	
};

}
#endif
