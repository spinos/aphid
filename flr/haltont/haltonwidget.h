/*
 *  haltonwidget.h
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef HALTON_WIDGET_H
#define HALTON_WIDGET_H

#include <qt/Plot1DWidget.h>

namespace aphid {
class UniformPlot1D; 
}

class HaltonWidget : public aphid::Plot1DWidget {

	Q_OBJECT
	
public:
	HaltonWidget(QWidget *parent = 0);
	virtual ~HaltonWidget();
	
protected:

private:
	float HaltonWidget::calcHalton(int i, int b);
	aphid::UniformPlot1D * m_trainPlot;
    
};

#endif
