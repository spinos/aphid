/*
 *  assetdlg.h
 *  wbg
 *
 *  list assets as tree
 *  Created by jian zhang on 3/21/17.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
 
#include <QDialog>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QPushButton;
class QTreeWidget;
class QTreeWidgetItem;
class QGraphicsScene;
class QGraphicsView;
class DeviationGraphics;
class DeviationLeaf;
class PreviewControl;
class QueryTexture;
QT_END_NAMESPACE

class AssetDlg : public QDialog
{
    Q_OBJECT

public:
    AssetDlg(QWidget *parent = 0);
	
	void keyPressEvent(QKeyEvent *e);

public slots:
	void onSelectATexture(QString texname);
	
private slots:
	
private:
	QTreeWidget * m_assetTree;
	
	
};