#include "BarbEdit.h"
#include <QtGui>
#include "BarbView.h"
BarbEdit::BarbEdit(QWidget *parent)
    : QDialog(parent)
{
	m_view = new BarbView(this);
	
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(m_view);
	layout->setStretch(0, 1);
	setLayout(layout);
	setWindowTitle(tr("Barb Preview"));
	
	setContentsMargins(0, 0, 0, 0);
	layout->setContentsMargins(0, 0, 0, 0);
	
	//connect(tools, SIGNAL(actionTriggered(int)), this, SLOT(receiveToolAction(int)));*/
}

QWidget * BarbEdit::barbView()
{
	return m_view;
}