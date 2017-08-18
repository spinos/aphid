/*
 *  PaletteView.cpp
 *  
 *
 *  Created by jian zhang on 3/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "PaletteView.h"
#include "graphchart/ShrubChartView.h"
#include "PiecesList.h"
#include <gar_common.h>
#include <boost/format.hpp>

using namespace aphid;

PaletteView::PaletteView(ShrubChartView* chart, QWidget *parent) : QWidget(parent)
{
	m_glyphList = new PiecesList(this);
	m_chart = chart;
	
	QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(m_glyphList);
	mainLayout->addWidget(m_chart);
	mainLayout->setContentsMargins(0,0,0,0);
	setLayout(mainLayout);
}

void PaletteView::showNamedPieces(const QString & swhat)
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
	else if(swhat == tr("allTwig") ) {
		m_glyphList->showTwigPieces();
	}
}
