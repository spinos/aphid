/*
 *  exportDlg.cpp
 *  garden
 *
 *  Created by jian zhang on 4/27/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "exportDlg.h"
#include "Vegetation.h"

using namespace aphid;

ExportDlg::ExportDlg(Vegetation * vege, QWidget *parent) : QDialog(parent)
{
	setWindowTitle(tr("Export Garden Example Statistics") );
	m_lab = new QLabel(this);
	m_lab->setFrameStyle(QFrame::Panel | QFrame::Sunken);
	m_lab->setAlignment(Qt::AlignTop | Qt::AlignLeft);
	m_applyBtn = new QPushButton(tr("Export"), this);
	m_cancelBtn = new QPushButton(tr("Cancel"), this);
	QHBoxLayout *btnLayout = new QHBoxLayout;
	btnLayout->addWidget(m_applyBtn);
	btnLayout->addWidget(m_cancelBtn);
	
	connect(m_applyBtn, SIGNAL(clicked()), 
			this, SLOT(setFilename()));
			
	connect(m_cancelBtn, SIGNAL(clicked()), 
			this, SLOT(reject()));

	fillLab(vege);
	
	QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(m_lab);
	mainLayout->addLayout(btnLayout);
	setLayout(mainLayout);
	resize(400, 300);
	
	m_filename = "";
}

void ExportDlg::fillLab(Vegetation * vege)
{
#if 0
	if(vege->numPatches() < 2) {
		m_lab->setText(tr("no example synthesized, cannot export"));
		m_applyBtn->setDisabled(true);
		return;
	}
#endif
	QString note(tr("synthesized n example "));
	QTextStream(&note)<<vege->numPatches();
	QTextStream(&note)<<"\n n instance "<<vege->getNumInstances();
	QTextStream(&note)<<"\n n geom "<<vege->numCachedGeoms();
#if 0
	std::string mshName;
	ATriangleMesh * mshVal = NULL;
	vege->geomBegin(mshName, mshVal);
	while(mshVal) {
		QTextStream(&note)<<"\n  "<<mshName.c_str();
		vege->geomNext(mshName, mshVal);
	}
#endif
	m_lab->setText(note);
	
}

void ExportDlg::setFilename()
{
	QString fileName = QFileDialog::getSaveFileName(this,
			tr("Export To File"), "~", tr("Garden Example Files (*.gde)"));
	
	if(fileName.length() < 5) {
		reject();
		return;
	}
	m_filename = fileName.toStdString();
	accept();
}

const std::string & ExportDlg::exportToFilename() const
{ return m_filename; }
