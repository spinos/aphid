/*
 *  Base2DView.h
 *  mallard
 *
 *  Created by jian zhang on 10/2/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Base3DView.h>

class Base2DView : public Base3DView {
	Q_OBJECT
public:
	Base2DView(QWidget *parent = 0);
	virtual ~Base2DView();
	
	virtual void processCamera(QMouseEvent *event);
private:
};
