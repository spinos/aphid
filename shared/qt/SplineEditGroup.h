/*
 *  SplineEditGroup.h
 *  
 *  spline edit with name
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SPLINE_EDIT_GRP_H
#define APH_SPLINE_EDIT_GRP_H

#include <QGroupBox>

QT_BEGIN_NAMESPACE
class QLabel;
QT_END_NAMESPACE

namespace aphid {

class QSplineEdit;

class SplineEditGroup : public QGroupBox 
{
	Q_OBJECT
public:
	SplineEditGroup(const QString& labelName,
		QWidget *parent = 0);
	void setNameId(int x);
	const int& nameId() const;
	
	void setSplineValue(const float* x);
	void setSplineCv0(const float* x);
	void setSplineCv1(const float* x);
	
signals:
	void valueChanged(QPair<int, QPointF> x);
	void leftControlChanged(QPair<int, QPointF> p);
	void rightControlChanged(QPair<int, QPointF> p);

private slots:
	void recvEditValue(QPointF p);
	void recvEditLleftControl(QPointF p);
	void recvEditRightControl(QPointF p);
	
private:
	QLabel* m_lab;
	QSplineEdit* m_edit;
	int m_nameId;
	
};

}

#endif
