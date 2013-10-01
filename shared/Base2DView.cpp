/*
 *  Base2DView.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/2/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "Base2DView.h"

#include <QtGui>

Base2DView::Base2DView(QWidget *parent) : Base3DView(parent)
{}

Base2DView::~Base2DView() {}

void Base2DView::processCamera(QMouseEvent *event)
{
    int dx = event->x() - lastMousePos().x();
    int dy = event->y() - lastMousePos().y();
    if (event->buttons() & Qt::LeftButton) {
        
    } 
	else if (event->buttons() & Qt::MidButton) {
		getCamera()->track(dx, dy);
    }
	else if (event->buttons() & Qt::RightButton) {
		getCamera()->zoom(-dx / 2 + -dy / 2);
		if(getCamera()->isOrthographic())
			updateOrthoProjection();
		else
			updatePerspProjection();
    }
}