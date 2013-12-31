#pragma once

#include <QDialog>
class BarbView;
class BarbControl;
class BarbEdit : public QDialog
{
    Q_OBJECT
	
public:
	BarbEdit(QWidget *parent = 0);
	
	QWidget * barbView();
	QWidget * barbControl();
signals:
	
public slots:
	
private:
	BarbView * m_view;
	BarbControl * m_control;
};
