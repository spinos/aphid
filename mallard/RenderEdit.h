/*
 *  RenderEdit.h
 *  mallard
 *
 *  Created by jian zhang on 12/29/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <QDialog>
class ImageView;
class RenderEdit : public QDialog
{
    Q_OBJECT
	
public:
	RenderEdit(QWidget *parent = 0);
	
signals:
	
public slots:
	
private:
	ImageView * m_view;
};