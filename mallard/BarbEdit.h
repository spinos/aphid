#pragma once

#include <QDialog>
class BarbView;
class BarbEdit : public QDialog
{
    Q_OBJECT
	
public:
	BarbEdit(QWidget *parent = 0);
	
signals:
	
public slots:
	
private:
	BarbView * m_view;
};
