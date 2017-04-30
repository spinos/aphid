/*
 *  RenderEdit.cpp
 *  mallard
 *
 *  Created by jian zhang on 12/29/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "RenderEdit.h"
#include "ImageView.h"

RenderEdit::RenderEdit(QWidget *parent)
    : QDialog(parent)
{
	m_view = new ImageView(this);
	
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(m_view);
	setLayout(layout);
	setWindowTitle(tr("Render View"));
	
	setContentsMargins(0, 0, 0, 0);
	layout->setContentsMargins(0, 0, 0, 0);
}

void RenderEdit::resizeRenderView(QSize s)
{
	m_view->resizeImage(s);
	resize(s);
}

void RenderEdit::setRenderEngine(QString name)
{
	m_view->setRendererName(name);
}

void RenderEdit::startRender(QString name)
{
	m_view->startRender(name);
}

void RenderEdit::keyPressEvent(QKeyEvent *e)
{
    if(e->key() == Qt::Key_Q) emit cancelRender();
	QDialog::keyPressEvent(e);
}

void RenderEdit::reject()
{
	emit cancelRender();
	QDialog::reject();
}
