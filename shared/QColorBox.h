/*
 *  QColorBox.h
 *  mallard
 *
 *  Created by jian zhang on 1/26/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <QWidget>

QT_BEGIN_NAMESPACE
class QColor;
QT_END_NAMESPACE

class QColorBox : public QWidget 
{
	Q_OBJECT
public:
	QColorBox(QWidget *parent = 0);
	
	void setColor(QColor c);
	QColor color() const;
	
public slots:
	void changeValue(int x);
	
protected:
	virtual void paintEvent( QPaintEvent * event );
	virtual void mousePressEvent(QMouseEvent *event);
	
private slots:
	
signals:
	void colorChanged(QColor c);
	
private:
	void chooseColor();
	QColor m_color;
};
