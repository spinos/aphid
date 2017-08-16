#ifndef APH_QENUM_COMBO_H
#define APH_QENUM_COMBO_H

#include <QWidget>

QT_BEGIN_NAMESPACE
class QLabel;
class QComboBox;
class QHBoxLayout;
QT_END_NAMESPACE

namespace aphid {

class QEnumCombo : public QWidget
{
    Q_OBJECT
public:
    QEnumCombo(const QString & name, QWidget * parent = 0);
	
	void addField(const QString & name,
				const int& val);
    
	void setNameId(int x);
	const int& nameId() const;
    
	void setValue(const int& x);
	int value();

protected slots:
    void sendEnumValue(int index);
    
signals:
    void valueChanged2(QPair<int, int> x);
	
protected:
    
private:
    QHBoxLayout* m_layout;
	QLabel * m_label;
	QComboBox * m_combo;
	int m_nameId;
};

}
#endif        //  #ifndef QEnumCombo_H

