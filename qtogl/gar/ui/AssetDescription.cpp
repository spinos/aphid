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
#include "data/billboard.h"
#include "data/variation.h"
#include "data/stem.h"
#include "data/twig.h"

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
		case gar::ggSprite :
			showSpriteGesc(tg);
		break;
		case gar::ggVariant :
			showVariantGesc(tg);
		break;
		case gar::ggStem :
			showStemGesc(tg);
		break;
		case gar::ggTwig :
			showTwigGesc(tg);
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

void AssetDescription::showSpriteGesc(const QPoint & tg)
{
	const int & billboardTyp = gar::ToBillboardType(tg.x() );
	m_lab->setText(tr(gar::BillboardTypeNames[billboardTyp]));
	QPixmap px(tr(gar::BillboardTypeImages[billboardTyp]) );
	m_pic->setPixmap(px);
	m_dtl->setText(tr(gar::BillboardTypeDescs[billboardTyp]));
}

void AssetDescription::showVariantGesc(const QPoint & tg)
{
	const int & variantTyp = gar::ToVariationType(tg.x() );
	m_lab->setText(tr(gar::VariationTypeNames[variantTyp]));
	QPixmap px(tr(gar::VariationTypeImages[variantTyp]) );
	m_pic->setPixmap(px);
	m_dtl->setText(tr(gar::VariationTypeDescs[variantTyp]));
}

void AssetDescription::showStemGesc(const QPoint & tg)
{
	const int & stemTyp = gar::ToStemType(tg.x() );
	m_lab->setText(tr(gar::StemTypeNames[stemTyp]));
	QPixmap px(tr(gar::StemTypeImages[stemTyp]) );
	m_pic->setPixmap(px);
	m_dtl->setText(tr(gar::StemTypeDescs[stemTyp]));
}

void AssetDescription::showTwigGesc(const QPoint & tg)
{
	const int & twigTyp = gar::ToTwigType(tg.x() );
	m_lab->setText(tr(gar::TwigTypeNames[twigTyp]));
	QPixmap px(tr(gar::TwigTypeImages[twigTyp]) );
	m_pic->setPixmap(px);
	m_dtl->setText(tr(gar::TwigTypeDescs[twigTyp]));
}
	