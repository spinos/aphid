/*
 *  HeightFieldAttrib.h
 *  
 *
 *  Created by jian zhang on 3/22/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef WBG_HEIGHT_FIELD_ATTRIB_H
#define WBG_HEIGHT_FIELD_ATTRIB_H

#include <QWidget>

QT_BEGIN_NAMESPACE
class QLineEdit;
QT_END_NAMESPACE

namespace aphid {
class IconLine;
class NavigatorWidget;
class ContextIconFrame;
}

class HeightFieldAttrib : public QWidget
{
	Q_OBJECT
	
public:
	HeightFieldAttrib(QWidget *parent = 0);
	
	void selHeightField(int idx);
	
protected:

signals:
	void tranformToolChanged(int x);

public slots:
	
private slots:
	void onTransformToolOn(int x);
	void onTransformToolOff(int x);

private:
	aphid::IconLine * m_fileNameLine;
	aphid::IconLine * m_imageSizeLine;
	aphid::NavigatorWidget * m_navigator;
	aphid::ContextIconFrame * m_transformToolIcons[3];
	
};
#endif