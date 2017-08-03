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
#include "gar_common.h"
#include "data/ground.h"
#include "data/grass.h"
#include "data/file.h"

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

void AssetDescription::recvAssetSel(QPoint tg)
{
	switch (tg.y() ) {
		case gar::ggGround :
			showGroundGesc(tg);
		break;
		case gar::ggGrass :
			showGrassGesc(tg);
		break;
		case gar::ggFile :
			showFileGesc(tg);
		break;
		default:
			qDebug()<<" AssetDescription::recvAssetSel group is unknown";
		;
	}
}
 
void AssetDescription::showGroundGesc(const QPoint & tg)
{
	const int & groundTyp = gar::ToGroundType(tg.x() );
	m_lab->setText(tr(gar::GroundTypeNames[groundTyp]));
	QPixmap px(tr(gar::GroundTypeImages[groundTyp]) );
	m_pic->setPixmap(px);
	m_dtl->setText(tr(gar::GroundTypeDescs[groundTyp]));
}
	
void AssetDescription::showGrassGesc(const QPoint & tg)
{
	const int & grassTyp = gar::ToGrassType(tg.x() );
	m_lab->setText(tr(gar::GrassTypeNames[grassTyp]));
	QPixmap px(tr(gar::GrassTypeImages[grassTyp]) );
	m_pic->setPixmap(px);
	m_dtl->setText(tr(gar::GrassTypeDescs[grassTyp]));
}

void AssetDescription::showFileGesc(const QPoint & tg)
{
	const int & fileTyp = gar::ToFileType(tg.x() );
	m_lab->setText(tr(gar::FileTypeNames[fileTyp]));
	QPixmap px(tr(gar::FileTypeImages[fileTyp]) );
	m_pic->setPixmap(px);
	m_dtl->setText(tr(gar::FileTypeDescs[fileTyp]));
}
