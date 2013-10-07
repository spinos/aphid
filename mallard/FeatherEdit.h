/*
 *  FeatherEdit.h
 *  mallard
 *
 *  Created by jian zhang on 10/2/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <QDialog>

class MlUVView;
class FeatherEdit : public QDialog
{
    Q_OBJECT
	
public:
	FeatherEdit(QWidget *parent = 0);
	
signals:
	void textureLoaded(QString name);
	
public slots:
	void receiveToolAction(int a);
	void receiveTexture(QString name);
private:
	MlUVView * m_view;
};