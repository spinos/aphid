/*
 *  FeatherEdit.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/2/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "FeatherEdit.h"
#include <QtGui>
#include <MlUVView.h>

FeatherEdit::FeatherEdit(QWidget *parent)
    : QDialog(parent)
{
	m_view = new MlUVView(this);
	
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(m_view);
	setLayout(layout);
	setWindowTitle(tr("Feather Editor"));
	
	setContentsMargins(0, 0, 0, 0);
	layout->setContentsMargins(0, 0, 0, 0);
}
