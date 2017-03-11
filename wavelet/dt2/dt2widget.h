/*
 *  dt2widget.h
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef DT_FT_2D_WIDGET_H
#define DT_FT_2D_WIDGET_H

#include <qt/Plot2DWidget.h>

class Dt2Widget : public aphid::Plot2DWidget {

	Q_OBJECT
	
public:
	Dt2Widget(QWidget *parent = 0);
	virtual ~Dt2Widget();
	
protected:

private:

};

#endif
