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
	void cancelRender();
public slots:
	void resizeRenderView(QSize s);
	void setRenderEngine(QString name);
	void startRender(QString name);
protected:
	virtual void keyPressEvent(QKeyEvent *e);
	virtual void reject();
private:
	ImageView * m_view;
};