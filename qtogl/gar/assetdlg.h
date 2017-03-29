/*
 *  assetdlg.h
 *  garden
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
QT_END_NAMESPACE

class GroundAssets;
class PlantAssets;
class GrassPalette;

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
	
public slots:
	
private slots:
	void onSelectAsset(QTreeWidgetItem * item, int column);
	
private:
	void lsGround();
	void lsPlant();
	
private:
	QTreeWidget * m_assetTree;
	GroundAssets * m_groundAsset;
	PlantAssets * m_plantAsset;
	QSplitter * m_split;
	QScrollArea * m_rgtArea;
	GrassPalette * m_grassPlt;
	
};
#endif