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
QT_END_NAMESPACE

class ShrubScene;
class GardenGlyph;

namespace gar {
class Attrib;
}

class PieceAttrib;

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
	void recvDoubleValue(QPair<int, double> x);
	void recvStringValue(QPair<int, QString> x);
	
private:
	void lsAttribs(GardenGlyph* g);
	void clearAttribs();
	void lsDefault(GardenGlyph* g);
	void lsAdded(GardenGlyph* g);
	void lsAttr(gar::Attrib* attr);
	QWidget* shoFltAttr(gar::Attrib* attr);
	QWidget* shoStrAttr(gar::Attrib* attr);
	
private:
	ShrubScene* m_scene;
	QVBoxLayout *mainLayout;
	QSpacerItem* m_lastStretch;
    QQueue<QWidget *> m_collWigs;
	PieceAttrib * m_attribs;
	
};

#endif