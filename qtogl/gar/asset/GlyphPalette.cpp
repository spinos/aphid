/*
 *  GlyphPalette.cpp
 *  
 *
 *  Created by jian zhang on 3/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "GlyphPalette.h"
#include "AssetDescription.h"
#include "PiecesList.h"
#include <qt/ContextIconFrame.h>
#include <gar_common.h>
#include <boost/format.hpp>

using namespace aphid;

GlyphPalette::GlyphPalette(QWidget *parent) : QWidget(parent)
{
	m_glyphList = new PiecesList(this);
	m_describ = new AssetDescription(this);
	
	QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(m_glyphList);
	mainLayout->addWidget(m_describ);
	mainLayout->setContentsMargins(0,0,0,0);
	setLayout(mainLayout);
	
	connect(m_glyphList, SIGNAL(itemClicked(QListWidgetItem *) ), 
		this, SLOT(selectAGrass(QListWidgetItem *) ) );
		
	connect(this, SIGNAL(onAssetSel(QPoint) ), 
		m_describ, SLOT(recvAssetSel(QPoint) ) );

}

void GlyphPalette::selectAGrass(QListWidgetItem * item)
{
	QVariant di = item->data(Qt::UserRole+1);
	emit onAssetSel(di.toPoint());
}

void GlyphPalette::showNamedPieces(const QString & swhat)
{
	if(swhat == tr("allGrass") ) {
		m_glyphList->showGrassPieces();
	}
	else if(swhat == tr("allGround") ) {
		m_glyphList->showGroundPieces();
	}
	else if(swhat == tr("allFile") ) {
		m_glyphList->showFilePieces();
	}
	else if(swhat == tr("allSprite") ) {
		m_glyphList->showSpritePieces();
	}
	else if(swhat == tr("allVariation") ) {
		m_glyphList->showVariationPieces();
	}
	else if(swhat == tr("allStem") ) {
		m_glyphList->showStemPieces();
	}
}
