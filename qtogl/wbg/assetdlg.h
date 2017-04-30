/*
 *  assetdlg.h
 *  wbg
 *
 *  list assets as tree
 *  Created by jian zhang on 3/21/17.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef WBG_ASSET_DLG_H
#define WBG_ASSET_DLG_H

#include <QDialog>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QPushButton;
class QSplitter;
class QScrollArea;
class QTreeWidget;
class QTreeWidgetItem;
class QScrollArea;
QT_END_NAMESPACE

class HeightFieldAssets;
class HeightFieldAttrib;

class AssetDlg : public QDialog
{
    Q_OBJECT

public:
    AssetDlg(QWidget *parent = 0);
	
	void keyPressEvent(QKeyEvent *e);
	
protected:
	virtual void 	closeEvent ( QCloseEvent * e );

signals:
	void onAssetDlgClose();
	void onHeightFieldAdd();
	void onHeightFieldSel(int x);
	void sendHeightFieldTransformTool(int x);
	
public slots:
	
private slots:
	void onSelectAsset(QTreeWidgetItem * item, int column);
	void onHeightFieldTransformToolChanged(int x);
	
private:
	void lsHeightField();
	void loadHeightField(QTreeWidgetItem * item);
	void selectHeightField(QTreeWidgetItem * item);
	
private:
	QSplitter * m_split;
	QTreeWidget * m_assetTree;
	QScrollArea * m_rgtArea;
	HeightFieldAssets * m_heightFieldAsset;
	HeightFieldAttrib * m_heightFieldAttr;
	
};
#endif