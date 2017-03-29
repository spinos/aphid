/*
 *  AssetDescription.cpp
 *  
 *
 *  Created by jian zhang on 3/30/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "AssetDescription.h"
#include <gar_common.h>
#include "data/grass.h"

AssetDescription::AssetDescription(QWidget *parent) : QWidget(parent)
{
	m_lab = new QLabel(this);
	m_pic = new QLabel(this);
	m_dtl = new QLabel(this);
	m_dtl->setFrameStyle(QFrame::Panel | QFrame::Sunken);
	m_dtl->setAlignment(Qt::AlignBottom | Qt::AlignLeft);
	QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(m_lab);
	mainLayout->addWidget(m_dtl);
	mainLayout->addWidget(m_pic);
	setLayout(mainLayout);
	
}

void AssetDescription::recvAssetSel(int x)
{
	m_lab->setText(tr(gar::GrassTypeNames[x]));
	QPixmap px(tr(gar::GrassTypeImages[x]) );
	m_pic->setPixmap(px);
	m_dtl->setText(tr(gar::GrassTypeDescs[x]));
	
}
 



