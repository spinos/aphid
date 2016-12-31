/*
 *  QDoubleEdit.h
 *  mallard
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "QModelEdit.h"

namespace aphid {

class QDoubleEdit : public QModelEdit
{
    Q_OBJECT
public:
    QDoubleEdit(const QModelIndex & idx, QWidget * parent = 0);
    void setValue(double x);
	double value();

public slots:
    
signals:
    
protected:
    
private:
	QDoubleValidator m_validate;
    double m_value;
};

}