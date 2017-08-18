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
class QSplitter;
class QTreeWidget;
class QTreeWidgetItem;
QT_END_NAMESPACE

class ShrubChartView;
class PaletteView;

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
	void onSelectAsset(QTreeWidgetItem * item, int column);
	
private:
	void lsAssets();
	
private:
	PaletteView * m_chartView;
	QTreeWidget * m_assetTree;
	QSplitter * m_split;
};
#endif