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
#include "data/ground.h"
#include "data/grass.h"
#include "data/file.h"

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
	lsDefault(g);
	switch(g->glyphType() ) {
		case gar::gtPot :
			break;
		case gar::gtBush :
			break;
		case gar::gtImportGeom :
			break;
		default:
			;
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

void AttribDlg::lsDefault(GardenGlyph* g)
{
	const int& gt = g->glyphType();
	QString stype(tr("unknown"));
	const int gg = gar::ToGroupType(gt );
	switch (gg) {
		case gar::ggGround :
			stype = gar::GroundTypeNames[gar::ToGroundType(gt)];
		break;
		case gar::ggGrass :
			stype = gar::GrassTypeNames[gar::ToGrassType(gt)];
		break;
		case gar::ggFile :
			stype = gar::FileTypeNames[gar::ToFileType(gt)];
		break;
		default:
		;
	}
	m_collWigs.enqueue(new QLabel(stype) );
}
