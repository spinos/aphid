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
#include "ShrubScene.h"
#include "graphchart/GardenGlyph.h"
#include "gar_common.h"

AttribDlg::AttribDlg(ShrubScene* scene, QWidget *parent) : QDialog(parent)
{
	m_scene = scene;
	setWindowTitle(tr("Attributes") );
	
	mainLayout = new QVBoxLayout;
    setLayout(mainLayout);
	resize(360, 240);
	
}

void AttribDlg::closeEvent ( QCloseEvent * e )
{
	emit onAttribDlgClose();
	QDialog::closeEvent(e);
}

void AttribDlg::recvSelectGlyph(bool x)
{
	clearAttribs();
	if(x) {
		GardenGlyph* g = m_scene->lastSelectedGlyph();
		setWindowTitle(tr("Attributes: %1").arg(g->glyphName().c_str() ) );
		lsAttribs(g);
			
	} else {
		setWindowTitle(tr("Attributes") );
	}
}

void AttribDlg::lsAttribs(GardenGlyph* g)
{
	switch(g->glyphType() ) {
		case gar::gtPot :
			break;
		case gar::gtBush :
			break;
		case gar::gtImportGeom :
			break;
		default:
			lsDefault(g);
	}
	
	
    
	const int n = m_collWigs.count();
    for (int i = 0; i < n; ++i) {
        mainLayout->addWidget(m_collWigs[i]);
    }

}

void AttribDlg::clearAttribs()
{
	foreach (QWidget *widget, m_collWigs)
        mainLayout->removeWidget(widget);

	while (!m_collWigs.isEmpty())
		delete m_collWigs.dequeue();
}

void AttribDlg::lsDefault()
{
	m_collWigs.enqueue(new QLabel);
}
