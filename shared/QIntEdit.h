/*
 *  QIntEdit.h
 *  mallard
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "QModelEdit.h"

class QIntEdit : public QModelEdit
{
    Q_OBJECT
public:
    QIntEdit(const QModelIndex & idx, QWidget * parent = 0);
    void setValue(int x);
	int value();

public slots:
    
signals:
    
protected:
    
private:
	QIntValidator m_validate;
    int m_value;
};