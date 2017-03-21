#include <QtGui>

#include "IconLine.h"

namespace aphid {

IconLine::IconLine(QWidget *parent)
    : QWidget (parent)
{
	QHBoxLayout * mainLayout = new QHBoxLayout;
	
	m_label = new QLabel(this);
	m_label->setMinimumWidth(64);
	m_label->setAlignment(Qt::AlignBottom | Qt::AlignRight);
	m_line = new QLineEdit(this);
	m_line->setMinimumWidth(200);
	
	mainLayout->addWidget(m_label);
	mainLayout->addWidget(m_line);
	mainLayout->setContentsMargins(0,0,0,0);
	setLayout(mainLayout);
	setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
}

void IconLine::setIconFile(const QString & fileName)
{
	QPixmap aPix(fileName);
	m_label->setPixmap(aPix);
}

void IconLine::setIconText(const QString & x)
{
	m_label->setText(x);
}

void IconLine::setLineText(const QString & x)
{
	m_line->setText(x);
}

}
