/*
 *  ContextToolGroup.cpp
 *  
 *
 *  Created by jian zhang on 9/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "ContextToolGroup.h"
#include "QIconFrame.h"

namespace aphid {

ContextToolProfile::ContextToolProfile()
{}

ContextToolProfile::~ContextToolProfile()
{}

void ContextToolProfile::setTitle(const QString& x)
{ m_title = x; }

void ContextToolProfile::addTool(int toolName, const QString& iconName0,
					const QString& iconName1)
{
	AToolContext actx;
	actx._name = toolName;
	actx._icon[0] = iconName0;
	actx._icon[1] = iconName1;
	m_ctx<<actx;
}
	
const QString& ContextToolProfile::title() const
{ return m_title; }

int ContextToolProfile::numTools() const
{ return m_ctx.count(); }

const ContextToolProfile::AToolContext& ContextToolProfile::getTool(
							int& row, int& col,
							const int& i) const
{
	row = i / 5;
	col = i - row * 5;
	return m_ctx[i]; 
}

int ContextToolProfile::numCols() const
{ return 5; }


ContextToolGroup::ContextToolGroup(ContextToolProfile* prof,
						QWidget *parent)
    : QGroupBox(parent)
{
	QGridLayout *grd = new QGridLayout;
	
	const int n = prof->numTools();
	for(int i=0;i<n;++i) {
		int row, col;
		const ContextToolProfile::AToolContext& it = prof->getTool(row, col, i);
		
		QIconFrame* frm = new QIconFrame;
		frm->setNameId(it._name);
		frm->addIconFile(it._icon[0]);
		frm->addIconFile(it._icon[1]);
		frm->setIconIndex(0);
		grd->addWidget(frm, row, col);
		
		m_tools<<frm;
		
		connect(frm, SIGNAL(iconChanged2(QPair<int, int>)),
				this, SLOT(recvContextChange(QPair<int, int>)));
	
	}
     
	grd->setColumnStretch(prof->numCols() - 1, 1);
	grd->setContentsMargins(1, 1, 1, 1);
	setLayout(grd);
	QSizePolicy sp(QSizePolicy::Minimum, QSizePolicy::Minimum);
	setSizePolicy(sp);
	setTitle(prof->title() );
		
}

void ContextToolGroup::recvContextChange(QPair<int, int> x)
{
	foreach(QIconFrame* its_, m_tools) {
		if(its_->nameId() != x.first)
			its_->setIconIndex(0);
	}

	QPair<int, int> val;
	val.first = x.first;
	val.second = x.second;
	
	emit toolSelected2(val);
}

void ContextToolGroup::setNameId(int x)
{ m_nameId = x; }

const int& ContextToolGroup::nameId() const
{ return m_nameId; }

}