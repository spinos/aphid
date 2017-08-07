#include <QtGui>
#include "QStringEditField.h"

namespace aphid {
    
QStringEditField::QStringEditField(const QString & name, QWidget * parent)
{
    m_label = new QLabel(name);
	m_label->setMinimumWidth(100);
	m_edit = new QLineEdit;
	m_edit->setMinimumWidth(80);
	
	m_layout = new QHBoxLayout;
	m_layout->addWidget(m_label);
	m_layout->addWidget(m_edit);
	m_layout->setStretch(1, 1);
	m_layout->setContentsMargins(0, 0, 0, 0);
	setLayout(m_layout);
	
	connect(m_edit, SIGNAL(returnPressed()),
            this, SLOT(sendEditValue()));
    
    m_selectFileFilter = "*";
}

void QStringEditField::setValue(const QString& x)
{ m_edit->setText(x); }

QString QStringEditField::value()
{ return m_edit->text(); }

void QStringEditField::setNameId(int x)
{ m_nameId = x; }

const int& QStringEditField::nameId() const
{ return m_nameId; }

void QStringEditField::addButton(const QString & iconName)
{
    m_button = new QPushButton;
    m_button->setIcon(QIcon(iconName));
    m_layout->addWidget(m_button);
    
    connect(m_button, SIGNAL(pressed()),
            this, SLOT(pickFile()));
    
}

void QStringEditField::pickFile()
{
	QString fileName = QFileDialog::getOpenFileName(this,
							tr("Choose a file to open"),
							"", m_selectFileFilter);
	if(fileName.size() > 4 ) {
	    m_edit->setText(fileName);
	    sendEditValue();
	}
}

void QStringEditField::sendEditValue()
{
    QPair<int, QString > val;
    val.first = m_nameId;
    val.second = m_edit->text();
    emit valueChanged2(val);
}

void QStringEditField::setSelectFileFilter(const QString & x)
{ m_selectFileFilter = x; }

}
