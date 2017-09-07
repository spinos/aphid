/*
 *  ToolDlg.cpp
 *  slerp
 *
 *  Created by jian zhang on 4/27/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "ToolDlg.h"
#include <qt/ContextToolGroup.h>
#include "slerp_common.h"

using namespace aphid;

ToolDlg::ToolDlg(QWidget *parent) : QDialog(parent)
{
	ContextToolProfile prof;
	prof.setTitle("Curve Tools");
	prof.addTool(slp::ctNew, ":icons/curve_new_inactive.png",
					":icons/curve_new.png");
	prof.addTool(slp::ctMove, ":icons/curve_move_inactive.png",
					":icons/curve_move.png");
	prof.addTool(slp::ctMoveStrand, ":icons/curve_move_strand_inactive.png",
					":icons/curve_move_strand.png");
	prof.addTool(slp::ctRotateStrand, ":icons/curve_rotate_strand_inactive.png",
					":icons/curve_rotate_strand.png");
	m_toolbox = new ContextToolGroup(&prof, this);
	
	QVBoxLayout* mainLayout = new QVBoxLayout;
	mainLayout->addWidget(m_toolbox);
	mainLayout->addStretch(1);
	mainLayout->setContentsMargins(8,8,8,8);
	setLayout(mainLayout);
	
	setWindowTitle(tr("Tools") );
	
	connect(m_toolbox, SIGNAL(toolSelected2(QPair<int, int>)),
				this, SLOT(recvToolSelected(QPair<int, int>)));
	
}

void ToolDlg::closeEvent ( QCloseEvent * e )
{
	emit onToolDlgClose();
	QDialog::closeEvent(e);
}

void ToolDlg::recvToolSelected(QPair<int, int> x)
{ 
	if(x.second)
		emit toolSelected(x.first); 
	else
		emit toolSelected(slp::ctUnknown); 
}
