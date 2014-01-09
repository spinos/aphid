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
	connect(m_control, SIGNAL(seedChanged(int)), m_view, SLOT(receiveSeed(int)));
	connect(m_control, SIGNAL(numSeparateChanged(int)), m_view, SLOT(receiveNumSeparate(int)));
	connect(m_control, SIGNAL(separateStrengthChanged(double)), m_view, SLOT(receiveSeparateStrength(double)));
	connect(m_control, SIGNAL(fuzzinessChanged(double)), m_view, SLOT(receiveFuzzy(double)));
	connect(m_control, SIGNAL(resShaftChanged(int)), m_view, SLOT(receiveResShaft(int)));
	connect(m_control, SIGNAL(resBarbChanged(int)), m_view, SLOT(receiveResBarb(int)));
	connect(m_control, SIGNAL(lodChanged(double)), m_view, SLOT(receiveLod(double)));
	
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