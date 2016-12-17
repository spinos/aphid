/*
 *  interpWidget.h
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef INTERP_WIDGET_H
#define INTERP_WIDGET_H

#include <Plot1DWidget.h>

namespace aphid {
class UniformPlot1D; 

namespace gpr {
class Interpolate1D;   
}
}

class InterpWidget : public aphid::Plot1DWidget {

	Q_OBJECT
	
public:
	InterpWidget(QWidget *parent = 0);
	virtual ~InterpWidget();
	
protected:
    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseMoveEvent(QMouseEvent *event);
	virtual void mouseReleaseEvent(QMouseEvent *event);
	
private:
    aphid::UniformPlot1D * m_trainPlot;
    aphid::UniformPlot1D * m_predictPlot;
    aphid::gpr::Interpolate1D * m_gpi;
    int m_selectedTrainInd;
    
};
#endif

