/*
 *  DictionaryView.cpp
 *  
 *
 *  Created by jian zhang on 9/9/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "DictionaryView.h"

namespace lfr {

DictionaryView::DictionaryView(QWidget *parent) : aphid::BaseImageWidget(parent)
{}

DictionaryView::~DictionaryView()
{}

void DictionaryView::recvDictionary(const QImage &image)
{
	m_pixmap = QPixmap::fromImage(image);
	update();
}

void DictionaryView::clientDraw(QPainter * pr)
{
	if (m_pixmap.isNull()) 
		return;
		
	pr->drawPixmap(QPoint(0,0), m_pixmap);
}

}