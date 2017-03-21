/*
 *  HeightFieldAttrib.cpp
 *  
 *
 *  Created by jian zhang on 3/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "HeightFieldAttrib.h"
#include <img/HeightField.h>
#include <qt/IconLine.h>
#include <ttg/GlobalElevation.h>
#include <qt/NavigatorWidget.h>
#include <qt/ContextIconFrame.h>
#include <wbg_common.h>
#include <boost/format.hpp>

using namespace aphid;

HeightFieldAttrib::HeightFieldAttrib(QWidget *parent) : QWidget(parent)
{
	m_fileNameLine = new IconLine(this);
	m_fileNameLine->setIconFile(tr(":/icons/heightField.png"));
	
	m_imageSizeLine = new IconLine(this);
	m_imageSizeLine->setIconText(tr("image size"));
	
	m_navigator = new NavigatorWidget(this);
	
	ContextIconFrame * moveAct = new ContextIconFrame(this);
	moveAct->addIconFile(":/icons/move_2d_inactive.png");
	moveAct->addIconFile(":/icons/move_2d.png");
	moveAct->setIconIndex(0);
	moveAct->setContext(wbg::hfcMove);
	
	ContextIconFrame * rotaAct = new ContextIconFrame(this);
	rotaAct->addIconFile(":/icons/rotate_2d_inactive.png");
	rotaAct->addIconFile(":/icons/rotate_2d.png");
	rotaAct->setIconIndex(0);
	rotaAct->setContext(wbg::hfcRotate);
	
	ContextIconFrame * scalAct = new ContextIconFrame(this);
	scalAct->addIconFile(":/icons/resize_2d_inactive.png");
	scalAct->addIconFile(":/icons/resize_2d.png");
	scalAct->setIconIndex(0);
	scalAct->setContext(wbg::hfcResize);
	
	m_transformToolIcons[0] = moveAct;
	m_transformToolIcons[1] = rotaAct;
	m_transformToolIcons[2] = scalAct;
	
	QHBoxLayout * toolLayout = new QHBoxLayout;
	toolLayout->addWidget(moveAct);
	toolLayout->addWidget(rotaAct);
	toolLayout->addWidget(scalAct);
	toolLayout->addStretch(8);
	
	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addLayout(toolLayout);
	mainLayout->addWidget(m_fileNameLine);
	mainLayout->addWidget(m_imageSizeLine);
	mainLayout->addWidget(m_navigator);
	mainLayout->addStretch(8);
	mainLayout->setContentsMargins(2,2,2,2);
	setLayout(mainLayout);
	
	for(int i=0;i<3;++i) {
		connect(m_transformToolIcons[i], SIGNAL(contextEnabled(int)), 
				this, SLOT(onTransformToolOn(int)));
		connect(m_transformToolIcons[i], SIGNAL(contextDisabled(int)), 
				this, SLOT(onTransformToolOff(int)));
	}
}

void HeightFieldAttrib::selHeightField(int idx)
{
	const img::HeightField & fld = ttg::GlobalElevation::GetHeightField(idx);
	m_fileNameLine->setLineText(tr(fld.fileName().c_str()) );
	const Int2 sz = fld.levelSignalSize(0);
	std::string ssz = boost::str(boost::format("%1% x %2%") % sz.x % sz.y );
		
	m_imageSizeLine->setLineText(tr(ssz.c_str()) );
	
	const int & l = fld.numLevels();
	int useLevel = l - 1;
	
	for(int i=0;i<l;++i) {
		const Array3<float> & sig = fld.levelSignal(i);
		if(sig.numCols() <= 256
			&& sig.numRows() <= 256) {
			useLevel = i;
			break;
		}
	}
	
	m_navigator->setImage(fld.levelSignal(useLevel));
	update();
}

void HeightFieldAttrib::onTransformToolOn(int x)
{
	for(int i=0;i<3;++i) {
		ContextIconFrame * frm = m_transformToolIcons[i];
		if(frm->getContext() != x) {
			frm->setIconIndex(0);
		}
	}
	
	emit tranformToolChanged(x);
}

void HeightFieldAttrib::onTransformToolOff(int x)
{
	emit tranformToolChanged(0);
}
