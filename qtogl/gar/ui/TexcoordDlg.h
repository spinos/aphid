/*
 *  TexcoordDlg.h
 *  garden
 *
 *  Created by jian zhang on 4/27/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_TEXCOORD_DLG_H
#define GAR_TEXCOORD_DLG_H

#include <QDialog>
#include <QQueue>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QPushButton;
class QVBoxLayout;
class QSpacerItem;
class QScrollArea;
QT_END_NAMESPACE

class ShrubScene;
class TexcoordWidget;

class TexcoordDlg : public QDialog
{
    Q_OBJECT

public:
    TexcoordDlg(ShrubScene* scene, QWidget *parent = 0);
	
protected:
	virtual void 	closeEvent ( QCloseEvent * e );

signals:
	void onTexcoordDlgClose();
		
public slots:
/// on select glyph
/// true: update attribs
/// false: clear attribs
	void recvSelectGlyph(bool x);
	
	QWidget* getWidget();

private:
	TexcoordWidget* m_widget;
	
};

#endif