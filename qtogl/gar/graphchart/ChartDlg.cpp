/*
 *  ChartDlg.cpp
 *  garden
 *
 *  Created by jian zhang on 8/4/17.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
 
#include <QtGui>
#include "ChartDlg.h"
#include "ShrubChartView.h"

using namespace std;

ChartDlg::ChartDlg(ShrubChartView* chart, QWidget *parent)
    : QDialog(parent)
{	
	m_chart = chart;
	
    QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->setContentsMargins(0,0,0,0);
    mainLayout->addWidget(m_chart);

    setLayout(mainLayout);
	setWindowTitle(tr("Graph View") );
	
}

void ChartDlg::keyPressEvent(QKeyEvent *e)
{}

void ChartDlg::closeEvent ( QCloseEvent * e )
{
	emit onChartDlgClose();
	QDialog::closeEvent(e);
}
