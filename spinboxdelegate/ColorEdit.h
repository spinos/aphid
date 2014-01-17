/*
 *  ColorEdit.h
 *  spinboxdelegate
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <QtGui>

class ColorEdit : public QWidget
{
    Q_OBJECT
public:
    ColorEdit(QWidget * parent = 0);
    void setColor(QColor color);
	QColor color() const;
    
public slots:
    
signals:
    
protected:
    virtual void mousePressEvent(QMouseEvent *event);
private:
    QFrame *m_button;
    QColor m_color;
};