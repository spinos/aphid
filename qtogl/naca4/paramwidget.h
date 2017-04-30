/*
 *  paramwidget.h
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef NACA_4_PARAM_WIDGET_H
#define NACA_4_PARAM_WIDGET_H

namespace aphid {

class QDoubleEditSlider;

}

#include <QWidget>

class ParamWidget : public QWidget {

	Q_OBJECT
	
public:
	ParamWidget(QWidget *parent = 0);
	virtual ~ParamWidget();
	
signals:
	void camberChanged(double x);
	void positionChanged(double x);
	void thicknessChanged(double x);
	
private slots:
	void sendCamber(double x);
	void sendPosition(double x);
	void sendThickness(double x);
	
private:
	aphid::QDoubleEditSlider * m_camberEdit;
	aphid::QDoubleEditSlider * m_positionEdit;
	aphid::QDoubleEditSlider * m_thicknessEdit;

};
#endif
