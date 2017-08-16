#include <QtGui>
#include "QEnumCombo.h"

namespace aphid {
    
QEnumCombo::QEnumCombo(const QString & name, QWidget * parent)
{
    m_label = new QLabel(name);
	m_label->setMinimumWidth(100);
	m_combo = new QComboBox;
	m_combo->setMinimumWidth(80);
	
	m_layout = new QHBoxLayout;
	m_layout->addWidget(m_label);
	m_layout->addWidget(m_combo);
	m_layout->setStretch(1, 1);
	m_layout->setContentsMargins(0, 0, 0, 0);
	setLayout(m_layout);
	
	connect(m_combo, SIGNAL(currentIndexChanged(int)),
            this, SLOT(sendEnumValue(int)));
    
}

void QEnumCombo::addField(const QString & name,
				const int& val)
{
	m_combo->addItem(name, QVariant(val) );
}

void QEnumCombo::sendEnumValue(int index)
{
    QPair<int, int > val;
    val.first = m_nameId;
    val.second = m_combo->itemData(index).toInt();
    emit valueChanged2(val);
}

void QEnumCombo::setValue(const int& x)
{ 
	int ind = 0;
	const int n = m_combo->count();
	for(int i=0;i<n;++i) {
		if(m_combo->itemData(i).toInt() == x) {
			ind = i;
			break;
		}
	}
	m_combo->setCurrentIndex(ind); 
}

int QEnumCombo::value()
{ 
	int i = m_combo->currentIndex();
	return m_combo->itemData(i).toInt(); 
}

void QEnumCombo::setNameId(int x)
{ m_nameId = x; }

const int& QEnumCombo::nameId() const
{ return m_nameId; }

}
