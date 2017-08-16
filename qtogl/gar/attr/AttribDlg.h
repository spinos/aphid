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
class StringAttrib;
class SplineAttrib;
class EnumAttrib;
}

class AttribDlg : public QDialog
{
    Q_OBJECT

public:
    AttribDlg(ShrubScene* scene, QWidget *parent = 0);
	
protected:
	virtual void 	closeEvent ( QCloseEvent * e );

signals:
	void onAttribDlgClose();
	void sendAttribChanged();
		
public slots:
/// on select glyph
/// true: update attribs
/// false: clear attribs
	void recvSelectGlyph(bool x);
	
private slots:
	void recvDoubleValue(QPair<int, double> x);
	void recvIntValue(QPair<int, int> x);
	void recvStringValue(QPair<int, QString> x);
	void recvSplineValue(QPair<int, QPointF> x);
	void recvSplineCv0(QPair<int, QPointF> x);
	void recvSplineCv1(QPair<int, QPointF> x);
	void recvEnumValue(QPair<int, int> x);
	
private:
	void lsAttribs(GardenGlyph* g);
	void clearAttribs();
	void lsDefault(GardenGlyph* g);
	void lsAdded(GardenGlyph* g);
	void lsAttr(gar::Attrib* attr);
	QWidget* shoFltAttr(gar::Attrib* attr);
	QWidget* shoIntAttr(gar::Attrib* attr);
	QWidget* shoStrAttr(gar::Attrib* attr);
	QWidget* shoSplineAttr(gar::Attrib* attr);
	QWidget* shoEnumAttr(gar::Attrib* attr);
	gar::StringAttrib* findStringAttr(int i);
	gar::SplineAttrib* findSplineAttr(int i);
	gar::EnumAttrib* findEnumAttr(int i);
	void updateSelectedGlyph();

private:
	ShrubScene* m_scene;
	GardenGlyph* m_selectedGlyph;
	QVBoxLayout *mainLayout;
	QSpacerItem* m_lastStretch;
    QQueue<QWidget *> m_collWigs;
	
};

#endif