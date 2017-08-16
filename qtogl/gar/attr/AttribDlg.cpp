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
#include "data/billboard.h"
#include "data/variation.h"
#include "data/stem.h"
#include "attr/PieceAttrib.h"
#include <qt/QDoubleEditSlider.h>
#include <qt/QStringEditField.h>
#include <qt/SplineEditGroup.h>
#include <qt/QEnumCombo.h>
#include <qt/IntEditGroup.h>

using namespace aphid;

AttribDlg::AttribDlg(ShrubScene* scene, QWidget *parent) : QDialog(parent)
{
	m_selectedGlyph = NULL;
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
	if(x) {
		GardenGlyph* g = m_scene->lastSelectedGlyph();
		if(g == m_selectedGlyph) {
			return;
		}
		m_selectedGlyph = g;
		clearAttribs();
		setWindowTitle(tr("Attributes: %1").arg(g->glyphName().c_str() ) );
		lsAttribs(g);
			
	} else {
		m_selectedGlyph = NULL;
		clearAttribs();
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
		case gar::ggSprite :
			stype = gar::BillboardTypeNames[gar::ToBillboardType(gt)];
		break;
		case gar::ggVariant :
			stype = gar::VariationTypeNames[gar::ToVariationType(gt)];
		break;
		case gar::ggStem :
			stype = gar::StemTypeNames[gar::ToStemType(gt)];
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
}

void AttribDlg::lsAttr(gar::Attrib* attr)
{
	const gar::AttribType& atyp = attr->attrType();
	
	QWidget* wig = NULL;
	switch (atyp) {
		case gar::tFloat :
			wig = shoFltAttr(attr);
		break;
		case gar::tInt :
			wig = shoIntAttr(attr);
		break;
		case gar::tString :
			wig = shoStrAttr(attr);
		break;
		case gar::tSpline :
			wig = shoSplineAttr(attr);
		break;
		case gar::tEnum :
			wig = shoEnumAttr(attr);
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

QWidget* AttribDlg::shoIntAttr(gar::Attrib* attr)
{
	IntEditGroup* wig = new IntEditGroup(tr(attr->attrNameStr() ) );
	int val, val0, val1;
	attr->getValue(val);
	attr->getMin(val0);
	attr->getMax(val1);
	wig->setLimit(val0, val1);
	wig->setValue(val);
	wig->setNameId(attr->attrName() );
	
	connect(wig, SIGNAL(valueChanged2(QPair<int, int>)),
            this, SLOT(recvIntValue(QPair<int, int>)));
			
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

QWidget* AttribDlg::shoSplineAttr(gar::Attrib* attr)
{
	gar::SplineAttrib* sattr = static_cast<gar::SplineAttrib*> (attr);
	
	SplineEditGroup* wig = new SplineEditGroup(tr(attr->attrNameStr()) );
	wig->setNameId(attr->attrName() );
	
	float tmp[2];
	sattr->getSplineValue(tmp);
	wig->setSplineValue(tmp);
	sattr->getSplineCv0(tmp);
	wig->setSplineCv0(tmp);
	sattr->getSplineCv1(tmp);
	wig->setSplineCv1(tmp);
	
	connect(wig, SIGNAL(valueChanged(QPair<int, QPointF>)),
            this, SLOT(recvSplineValue(QPair<int, QPointF>)));
			
	connect(wig, SIGNAL(leftControlChanged(QPair<int, QPointF>)),
            this, SLOT(recvSplineCv0(QPair<int, QPointF>)));
			
	connect(wig, SIGNAL(rightControlChanged(QPair<int, QPointF>)),
            this, SLOT(recvSplineCv1(QPair<int, QPointF>)));
			
	return wig;
}

QWidget* AttribDlg::shoEnumAttr(gar::Attrib* attr)
{
	gar::EnumAttrib* sattr = static_cast<gar::EnumAttrib*> (attr);
	
	QEnumCombo* wig = new QEnumCombo(tr(attr->attrNameStr()) );
	wig->setNameId(attr->attrName() );
	
	const int& nf = sattr->numFields();
	for(int i=0;i<nf;++i) {
		const int& fi = sattr->getField(i);
		QString fnm(gar::Attrib::IntAsEnumFieldName(fi) );
		wig->addField(fnm, fi);
	}
	
	int val;
	sattr->getValue(val);
	wig->setValue(val);
	
	connect(wig, SIGNAL(valueChanged2(QPair<int, int>)),
            this, SLOT(recvEnumValue(QPair<int, int>)));
	
	return wig;
}

void AttribDlg::recvDoubleValue(QPair<int, double> x)
{
	PieceAttrib* att = m_selectedGlyph->attrib();
	gar::Attrib* dst = att->findAttrib(gar::Attrib::IntAsAttribName(x.first) );	
	if(!dst) {
		qDebug()<<" AttribDlg::recvDoubleValue cannot find float attr "
			<<x.first;
		return;
	}
	dst->setValue((float)x.second);
	updateSelectedGlyph();
}

void AttribDlg::recvIntValue(QPair<int, int> x)
{
	PieceAttrib* att = m_selectedGlyph->attrib();
	gar::Attrib* dst = att->findAttrib(gar::Attrib::IntAsAttribName(x.first) );	
	if(!dst) {
		qDebug()<<" AttribDlg::recvDoubleValue cannot find int attr "
			<<x.first;
		return;
	}
	dst->setValue(x.second);
	updateSelectedGlyph();
}

void AttribDlg::recvStringValue(QPair<int, QString> x)
{
	gar::StringAttrib* sattr = findStringAttr(x.first);
	if(!sattr) 
		return;
	sattr->setValue(x.second.toStdString() );
	updateSelectedGlyph();
}

gar::StringAttrib* AttribDlg::findStringAttr(int i)
{
	if(!m_selectedGlyph) 
		return NULL;
		
	PieceAttrib* att = m_selectedGlyph->attrib();
	gar::Attrib* dst = att->findAttrib(gar::Attrib::IntAsAttribName(i) );	
	if(!dst) {
		qDebug()<<" AttribDlg cannot find string attr "
			<<i;
		return NULL;
	}
	
	return static_cast<gar::StringAttrib*> (dst);
}

gar::SplineAttrib* AttribDlg::findSplineAttr(int i)
{
	if(!m_selectedGlyph) 
		return NULL;
		
	PieceAttrib* att = m_selectedGlyph->attrib();
	gar::Attrib* dst = att->findAttrib(gar::Attrib::IntAsAttribName(i) );	
	if(!dst) {
		qDebug()<<" AttribDlg cannot find spline attr "
			<<i;
		return NULL;
	}
	
	return static_cast<gar::SplineAttrib*> (dst);
}

gar::EnumAttrib* AttribDlg::findEnumAttr(int i)
{
	if(!m_selectedGlyph) 
		return NULL;
		
	PieceAttrib* att = m_selectedGlyph->attrib();
	gar::Attrib* dst = att->findAttrib(gar::Attrib::IntAsAttribName(i) );	
	if(!dst) {
		qDebug()<<" AttribDlg cannot find enum attr "
			<<i;
		return NULL;
	}
	
	return static_cast<gar::EnumAttrib*> (dst);
}

void AttribDlg::recvSplineValue(QPair<int, QPointF> x)
{
	gar::SplineAttrib* sattr = findSplineAttr(x.first);
	if(!sattr) 
		return;
	
	float tmp[2];
	QPointF& p = x.second;
	tmp[0] = p.x();
	tmp[1] = p.y();
	sattr->setSplineValue(tmp[0], tmp[1] );
	updateSelectedGlyph();
}
	
void AttribDlg::recvSplineCv0(QPair<int, QPointF> x)
{
	gar::SplineAttrib* sattr = findSplineAttr(x.first);
	if(!sattr) 
		return;
		
	float tmp[2];
	QPointF& p = x.second;
	tmp[0] = p.x();
	tmp[1] = p.y();
	sattr->setSplineCv0(tmp[0], tmp[1]);
	updateSelectedGlyph();
}
	
void AttribDlg::recvSplineCv1(QPair<int, QPointF> x)
{
	gar::SplineAttrib* sattr = findSplineAttr(x.first);
	if(!sattr) 
		return;
		
	float tmp[2];
	QPointF& p = x.second;
	tmp[0] = p.x();
	tmp[1] = p.y();
	sattr->setSplineCv1(tmp[0], tmp[1]);
	updateSelectedGlyph();
}

void AttribDlg::updateSelectedGlyph()
{
	if(!m_selectedGlyph) 
		return;
		
	PieceAttrib* att = m_selectedGlyph->attrib();
	if(att->update())
		emit sendAttribChanged();
}

void AttribDlg::recvEnumValue(QPair<int, int> x)
{
	gar::EnumAttrib* sattr = findEnumAttr(x.first);
	if(!sattr) 
		return;
	
	sattr->setValue(x.second);
	updateSelectedGlyph();
}
