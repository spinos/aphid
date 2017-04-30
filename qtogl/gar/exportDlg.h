/*
 *  exportDlg.h
 *  garden
 *
 *  Created by jian zhang on 4/27/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_EXPORT_DLG_H
#define GAR_EXPORT_DLG_H

#include <QDialog>
#include <string>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QPushButton;
QT_END_NAMESPACE

class Vegetation;

class ExportDlg : public QDialog
{
    Q_OBJECT

public:
    ExportDlg(Vegetation * vege, QWidget *parent = 0);
	
protected:
		
public slots:
	void setFilename();
	
	const std::string & exportToFilename() const;
	
private slots:
	
private:
	void fillLab(Vegetation * vege);
	
private:
	QLabel * m_lab;
	QPushButton * m_applyBtn;
	QPushButton * m_cancelBtn;
	std::string m_filename;
	
};

#endif