/*
 *  linregwidget.h
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef LINREG_WIDGET_H
#define LINREG_WIDGET_H

#include <qt/Plot1DWidget.h>

namespace aphid {
class UniformPlot1D; 
}

class LinregWidget : public aphid::Plot1DWidget {

	Q_OBJECT
	
public:
	LinregWidget(QWidget *parent = 0);
	virtual ~LinregWidget();
	
protected:

private:
	
    aphid::UniformPlot1D * m_trainPlot;
    aphid::UniformPlot1D * m_predictPlot;
    aphid::UniformPlot1D * m_blendPlot;
    
};

#endif
