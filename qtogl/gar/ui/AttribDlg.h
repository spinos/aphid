/*
 *  AttribDlg.h
 *  garden
 *
 *  Created by jian zhang on 4/27/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_ATTRIB_DLG_H
#define GAR_ATTRIB_DLG_H

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
class AttribWidget;

class AttribDlg : public QDialog
{
    Q_OBJECT

public:
    AttribDlg(ShrubScene* scene, QWidget *parent = 0);
	
protected:
	virtual void 	closeEvent ( QCloseEvent * e );

signals:
	void onAttribDlgClose();
		
public slots:
/// on select glyph
/// true: update attribs
/// false: clear attribs
	void recvSelectGlyph(bool x);
	
	QWidget* getWidget();

private:
	QScrollArea* m_scroll;
	AttribWidget* m_widget;
	QLabel* m_lab;
	
};

#endif