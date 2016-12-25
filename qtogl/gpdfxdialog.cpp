#include <QtGui>
#include "gpdfxdialog.h"
#include "gpdfxwidget.h"

GpdfxDialog::GpdfxDialog(QWidget *parent)
    : QDialog(parent)
{	
	m_wig = new GpdfxWidget(this);
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(m_wig);
	layout->setStretch(2, 1);
	layout->setSpacing(4);
	
	setLayout(layout);
    setWindowTitle(tr("X input") );
    resize(480, 480);
}

void GpdfxDialog::recvXValue(QPointF x)
{ emit xValueChanged(x); }
