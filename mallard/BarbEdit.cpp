#include "BarbEdit.h"
#include <QtGui>
#include "BarbView.h"
#include "BarbControl.h"
BarbEdit::BarbEdit(QWidget *parent)
    : QDialog(parent)
{
	m_view = new BarbView(this);
	m_control = new BarbControl(this);
	
	QHBoxLayout *layout = new QHBoxLayout;
	layout->addWidget(m_view);
	layout->addWidget(m_control);
	layout->setStretch(0, 1);
	setLayout(layout);
	setWindowTitle(tr("Barb Preview"));
	
	setContentsMargins(0, 0, 0, 0);
	layout->setContentsMargins(0, 0, 0, 0);
	
	connect(m_control, SIGNAL(shapeChanged()), m_view, SLOT(receiveShapeChanged()));
}

BarbEdit::~BarbEdit()
{
	delete m_view;
	delete m_control;
}

QWidget * BarbEdit::barbView()
{
	return m_view;
}

QWidget * BarbEdit::barbControl()
{
	return m_control;
}