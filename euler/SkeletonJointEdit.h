#pragma once
#include <AllMath.h>
#include <QWidget>

class QDouble3Edit;
class SkeletonSystem;
class SkeletonJoint;

QT_BEGIN_NAMESPACE
class QGroupBox;
QT_END_NAMESPACE

class SkeletonJointEdit : public QWidget {
Q_OBJECT
public:
    SkeletonJointEdit(SkeletonSystem * skeleton, QWidget *parent = 0);
    virtual ~SkeletonJointEdit();
    
protected:

private:
	SkeletonSystem * m_skeleton;
    QGroupBox * controlGrp;
	QDouble3Edit * translation;
	QDouble3Edit * orientAngle;
	QDouble3Edit * rotateAngle;
	SkeletonJoint * m_activeJoint;
public slots:
    void attachToJoint(int idx);
	void setJointTranslation(Vector3F v);
	void setJointRotation(Vector3F v);
	void setJointOrient(Vector3F v);
	void updateValues();
	
signals:
	void valueChanged();
};
