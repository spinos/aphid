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
#include "attr/PieceAttrib.h"
#include <qt/QDoubleEditSlider.h>
#include <qt/QStringEditField.h>

using namespace aphid;

AttribDlg::AttribDlg(ShrubScene* scene, QWidget *parent) : QDialog(parent)
{
	m_attribs = NULL;
	m_scene = scene;
	setWindowTitle(tr("Attributes") );
	
	mainLayout = new QVBoxLayout;
	mainLayout->setSpacing(2);
    setLayout(mainLayout);
	resize(360, 240);
	m_lastStretch = 0;
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
	lsAdded(g);
    
	const int n = m_collWigs.count();
    for (int i = 0; i < n; ++i) {
        mainLayout->addWidget(m_collWigs[i]);
    }

	m_lastStretch = new QSpacerItem(8,8, QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
	mainLayout->addItem(m_lastStretch);
}

void AttribDlg::clearAttribs()
{
	foreach (QWidget *widget, m_collWigs)
        mainLayout->removeWidget(widget);

	while (!m_collWigs.isEmpty())
		delete m_collWigs.dequeue();
		
	if(m_lastStretch) {
		mainLayout->removeItem(m_lastStretch);
		delete m_lastStretch;
		m_lastStretch = 0;
	}
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
	m_collWigs.enqueue(new QLabel(tr(" node type: %1").arg(stype) ) );
}

void AttribDlg::lsAdded(GardenGlyph* g)
{
	PieceAttrib* att = g->attrib();
	const int n = att->numAttribs();
	for(int i=0;i<n;++i) {
		gar::Attrib* ati = att->getAttrib(i);
		lsAttr(ati);
	}
	m_attribs = att;
}

void AttribDlg::lsAttr(gar::Attrib* attr)
{
	const gar::AttribType& atyp = attr->attrType();
	
	QWidget* wig = NULL;
	switch (atyp) {
		case gar::tFloat :
			wig = shoFltAttr(attr);
		break;
		case gar::tString :
			wig = shoStrAttr(attr);
		break;
		default:
			wig = new QLabel(tr(attr->attrNameStr() ) );
	}
	
	m_collWigs.enqueue(wig);
}

QWidget* AttribDlg::shoFltAttr(gar::Attrib* attr)
{
	QDoubleEditSlider* wig = new QDoubleEditSlider(tr(attr->attrNameStr() ) );
	float val, val0, val1;
	attr->getValue(val);
	attr->getMin(val0);
	attr->getMax(val1);
	wig->setLimit(val0, val1);
	wig->setValue(val);
	wig->setNameId(attr->attrName() );
	
	connect(wig, SIGNAL(valueChanged2(QPair<int, double>)),
            this, SLOT(recvDoubleValue(QPair<int, double>)));
			
	return wig;
}

QWidget* AttribDlg::shoStrAttr(gar::Attrib* attr)
{
	gar::StringAttrib* sattr = static_cast<gar::StringAttrib*> (attr);
	std::string sval;
	sattr->getValue(sval);
	QStringEditField* wig = new QStringEditField(tr(attr->attrNameStr() ) );
	wig->setValue(tr(sval.c_str() ) );
	if(sattr->isFileName() ) {
	    wig->addButton(":/icons/document_open.png");
	    wig->setSelectFileFilter("*.hes;;*.m");
	}
	wig->setNameId(attr->attrName() );
	
	connect(wig, SIGNAL(valueChanged2(QPair<int, QString>)),
            this, SLOT(recvStringValue(QPair<int, QString>)));
    
	return wig;
}

void AttribDlg::recvDoubleValue(QPair<int, double> x)
{
	gar::Attrib* dst = m_attribs->findAttrib(PieceAttrib::IntAsAttribName(x.first) );	
	if(!dst) {
		qDebug()<<" AttribDlg::recvDoubleValue cannot find float attr "
			<<x.first;
	}
	dst->setValue((float)x.second);
}

void AttribDlg::recvStringValue(QPair<int, QString> x)
{
    gar::Attrib* dst = m_attribs->findAttrib(PieceAttrib::IntAsAttribName(x.first) );	
	if(!dst) {
		qDebug()<<" AttribDlg::recvDoubleValue cannot find string attr "
			<<x.first;
	}
	gar::StringAttrib* sattr = static_cast<gar::StringAttrib*> (dst);
	sattr->setValue(x.second.toStdString() );
}
