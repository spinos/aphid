/*
 *  gpdfxwidget.h
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef GP_DF_X_WIDGET_H
#define GP_DF_X_WIDGET_H

#include <qt/Plot1DWidget.h>

class GpdfxWidget : public aphid::Plot1DWidget {

	Q_OBJECT
	
public:
	GpdfxWidget(QWidget *parent = 0);
	virtual ~GpdfxWidget();
	
signals:
	void xValueChanged(QPointF x);
	
protected:
	virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseMoveEvent(QMouseEvent *event);
	
private:
	aphid::UniformPlot1D * bp;
	
};
#endif
