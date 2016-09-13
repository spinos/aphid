/*
 *  ft1dWidget.h
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef FT_1D_WIDGET_H
#define FT_1D_WIDGET_H

#include <Plot2DWidget.h>

class Ft1dWidget : public aphid::Plot2DWidget {

	Q_OBJECT
	
public:
	Ft1dWidget(QWidget *parent = 0);
	virtual ~Ft1dWidget();
	
protected:

private:

};

#endif
