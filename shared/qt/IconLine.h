/*
 *  IconLine.h
 *  icon and line edit
 *
 */

#ifndef APH_ICON_LINE_H
#define APH_ICON_LINE_H

#include <QWidget>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
QT_END_NAMESPACE

namespace aphid {

class IconLine : public QWidget
{
    Q_OBJECT

public:
    IconLine(QWidget *parent = 0);
	
	void setIconFile(const QString & fileName);
	void setIconText(const QString & x);
	void setLineText(const QString & x);

signals:

public slots:

protected:
    
private:
    QLabel * m_label;
	QLineEdit * m_line;
	
};

}

#endif
