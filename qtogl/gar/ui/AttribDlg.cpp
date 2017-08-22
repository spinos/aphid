/*
 *  AttribDlg.cpp
 *  garden
 *
 *  Created by jian zhang on 4/27/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "AttribDlg.h"
#include "AttribWidget.h"
#include "graphchart/GardenGlyph.h"
#include "ShrubScene.h"

using namespace aphid;

AttribDlg::AttribDlg(ShrubScene* scene, QWidget *parent) : QDialog(parent)
{
	m_lab = new QLabel(tr(" node type: unknown"));
	m_widget = new AttribWidget(scene, this);
	setWindowTitle(tr("Attributes") );
	
	m_scroll = new QScrollArea;
	m_scroll->setWidgetResizable(true);
	
	m_scroll->setWidget(m_widget);
	
	QVBoxLayout* mainLayout = new QVBoxLayout;
	mainLayout->addWidget(m_lab);
	mainLayout->addWidget(m_scroll);
	mainLayout->setSpacing(2);
    mainLayout->setContentsMargins(8,8,8,8);
	setLayout(mainLayout);
	
	resize(360, 480);
	
}

void AttribDlg::closeEvent ( QCloseEvent * e )
{
	emit onAttribDlgClose();
	QDialog::closeEvent(e);
}

void AttribDlg::recvSelectGlyph(bool x)
{
	m_widget->recvSelectGlyph(x);
	if(x) {
		setWindowTitle(tr("Attributes: %1").arg(m_widget->lastSelectedGlyphName() ) );
		
	} else {
		setWindowTitle(tr("Attributes") );
	}
	QString stype = m_widget->lastSelectedGlyphTypeName();
	m_lab->setText(tr(" node type: %1").arg(stype) );	
}

QWidget* AttribDlg::getWidget()
{ return m_widget; }
