/*
 *  Ft2Widget.h
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef FT_2D_WIDGET_H
#define FT_2D_WIDGET_H

#include <qt/Plot2DWidget.h>

class Ft2Widget : public aphid::Plot2DWidget {

	Q_OBJECT
	
public:
	Ft2Widget(QWidget *parent = 0);
	virtual ~Ft2Widget();
	
protected:

private:

};

#endif
