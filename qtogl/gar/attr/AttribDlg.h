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
QT_END_NAMESPACE

class ShrubScene;
class GardenGlyph;

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
	
private slots:
	
private:
	void lsAttribs(GardenGlyph* g);
	void clearAttribs();
	void lsDefault();
	
private:
	ShrubScene* m_scene;
	QVBoxLayout *mainLayout;
    QQueue<QWidget *> m_collWigs;

};

#endif