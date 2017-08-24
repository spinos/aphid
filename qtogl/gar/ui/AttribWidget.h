/*
 *  AttribWidget.h
 *  
 *
 *  Created by jian zhang on 8/18/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_ATTRIB_WIDGET_H
#define GAR_ATTRIB_WIDGET_H

#include <QWidget>

class ShrubScene;
class GardenGlyph;

namespace gar {
class Attrib;
class StringAttrib;
class SplineAttrib;
class EnumAttrib;
class ActionAttrib;
}

class AttribWidget : public QWidget
{
    Q_OBJECT

public:
    AttribWidget(ShrubScene* scene, QWidget *parent = 0);
	
	void recvSelectGlyph(bool x);
	
	QString lastSelectedGlyphName() const;
	QString lastSelectedGlyphTypeName() const;

protected:

signals:
	void sendAttribChanged();
	
private slots:
	void recvDoubleValue(QPair<int, double> x);
	void recvIntValue(QPair<int, int> x);
	void recvStringValue(QPair<int, QString> x);
	void recvSplineValue(QPair<int, QPointF> x);
	void recvSplineCv0(QPair<int, QPointF> x);
	void recvSplineCv1(QPair<int, QPointF> x);
	void recvEnumValue(QPair<int, int> x);
	void recvActionPressed(QPair<int, int> x);
	void recvVec2Value(QPair<int, QVector<double> > x);
	
private:
	void lsAttribs(GardenGlyph* g);
	void clearAttribs();
	void lsAdded(GardenGlyph* g);
	void lsAttr(gar::Attrib* attr);
	QWidget* shoFltAttr(gar::Attrib* attr);
	QWidget* shoIntAttr(gar::Attrib* attr);
	QWidget* shoStrAttr(gar::Attrib* attr);
	QWidget* shoSplineAttr(gar::Attrib* attr);
	QWidget* shoEnumAttr(gar::Attrib* attr);
	QWidget* shoVec2Attr(gar::Attrib* attr);
	QWidget* shoActionAttr(gar::Attrib* attr);
	gar::StringAttrib* findStringAttr(int i);
	gar::SplineAttrib* findSplineAttr(int i);
	gar::EnumAttrib* findEnumAttr(int i);
	gar::ActionAttrib* findActionAttr(int i);
	void updateSelectedGlyph();

private:
	ShrubScene* m_scene;
	GardenGlyph* m_selectedGlyph;
	QVBoxLayout *mainLayout;
	QSpacerItem* m_lastStretch;
    QQueue<QWidget *> m_collWigs;
	
};

#endif
