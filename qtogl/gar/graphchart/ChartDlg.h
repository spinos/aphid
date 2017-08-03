/*
 *  ChartDlg.h
 *  garden
 *
 *  holds the node connection graph chart view
 *
 *  Created by jian zhang on 8/4/17.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef GAR_CHART_DLG_H
#define GAR_CHART_DLG_H

#include <QDialog>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class ShrubChartView;

class ChartDlg : public QDialog
{
    Q_OBJECT

public:
    ChartDlg(ShrubChartView* chart, QWidget *parent = 0);
	
	void keyPressEvent(QKeyEvent *e);
	
protected:
	virtual void 	closeEvent ( QCloseEvent * e );

signals:
	void onChartDlgClose();
	
public slots:
	
private slots:
	
private:
	
private:
	ShrubChartView * m_chart;
	
};
#endif