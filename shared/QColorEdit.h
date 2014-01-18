/*
 *  QColorEdit.h
 *  spinboxdelegate
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "QModelEdit.h"

class QColorEdit : public QModelEdit
{
    Q_OBJECT
public:
    QColorEdit(QColor color, const QModelIndex & idx, QWidget * parent = 0);
    void setColor(QColor color);
	QColor color() const;
    QColor pickColor();
public slots:
    
signals:
    
protected:
    
private:
    QColor m_color;
};