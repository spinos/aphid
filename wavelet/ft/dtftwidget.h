/*
 *  dtftwidget.h
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef DT_FT_WIDGET_H
#define DT_FT_WIDGET_H

#include <qt/Plot1DWidget.h>

class DtFtWidget : public aphid::Plot1DWidget {

	Q_OBJECT
	
public:
	DtFtWidget(QWidget *parent = 0);
	virtual ~DtFtWidget();
	
protected:

private:

};

#endif
