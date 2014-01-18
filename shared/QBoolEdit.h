/*
 *  QBoolEdit.h
 *  mallard
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "QModelEdit.h"

class QBoolEdit : public QModelEdit
{
    Q_OBJECT
public:
    QBoolEdit(const QModelIndex & idx, QWidget * parent = 0);
    void setValue(bool x);
	bool value();

public slots:
    
signals:
    
protected:
    
private:
	QString translateBoolToStr(bool src) const;
	bool translateStrToBool(const QString & src) const;
	bool m_value;
};