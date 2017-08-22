/*
 *  TexcoordDlg.cpp
 *  garden
 *
 *  Created by jian zhang on 4/27/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "TexcoordDlg.h"
#include "TexcoordWidget.h"
#include "graphchart/GardenGlyph.h"
#include "ShrubScene.h"

using namespace aphid;

TexcoordDlg::TexcoordDlg(ShrubScene* scene, QWidget *parent) : QDialog(parent)
{
	m_widget = new TexcoordWidget(scene, this);
	setWindowTitle(tr("UV") );
	
	QVBoxLayout* mainLayout = new QVBoxLayout;
	mainLayout->addWidget(m_widget);
	mainLayout->setContentsMargins(8,8,8,8);
	setLayout(mainLayout);
	
	resize(480, 480);
	
}

void TexcoordDlg::closeEvent ( QCloseEvent * e )
{
	emit onTexcoordDlgClose();
	QDialog::closeEvent(e);
}

void TexcoordDlg::recvSelectGlyph(bool x)
{
	m_widget->recvSelectGlyph(x);
}

QWidget* TexcoordDlg::getWidget()
{ return m_widget; }
