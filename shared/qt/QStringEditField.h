#ifndef APH_QSTRINGEDITFIELD_H
#define APH_QSTRINGEDITFIELD_H

#include <QWidget>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QHBoxLayout;
class QPushButton;
QT_END_NAMESPACE

namespace aphid {

class QStringEditField : public QWidget
{
    Q_OBJECT
public:
    QStringEditField(const QString & name, QWidget * parent = 0);
    void setNameId(int x);
	const int& nameId() const;
    void setValue(const QString& x);
	QString value();

	void addButton(const QString & iconName);
	void setSelectFileFilter(const QString & x);
	
protected slots:
    void pickFile();
    void sendEditValue();
    
signals:
    void valueChanged2(QPair<int, QString> x);
	
protected:
    
private:
    QHBoxLayout* m_layout;
	QLabel * m_label;
	QLineEdit * m_edit;
	QPushButton* m_button;
	QString m_selectFileFilter;
    int m_nameId;
};

}
#endif        //  #ifndef QSTRINGEDITFIELD_H

