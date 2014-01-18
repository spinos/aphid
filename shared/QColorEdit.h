/*
 *  QColorEdit.h
 *  spinboxdelegate
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <QtGui>

class QColorEdit : public QWidget
{
    Q_OBJECT
public:
    QColorEdit(QColor color, QWidget * parent = 0);
    void setColor(QColor color);
	QColor color() const;
    QColor pickColor();
public slots:
    
signals:
    
protected:
    
private:
    QFrame *m_button;
    QColor m_color;
};