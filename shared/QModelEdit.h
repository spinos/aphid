/*
 *  QModelEdit.h
 *  mallard
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <QtGui>

class QModelEdit : public QLineEdit
{
    Q_OBJECT
public:
    QModelEdit(const QModelIndex & idx, QWidget * parent = 0);
	QModelIndex index() const;
public slots:
    
signals:
    
protected:
    
private:
	QModelIndex m_index;
};