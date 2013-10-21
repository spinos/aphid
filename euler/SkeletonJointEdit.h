#pragma once

#include <QWidget>

QT_BEGIN_NAMESPACE
class QGroupBox;
QT_END_NAMESPACE

class SkeletonJointEdit : public QWidget {
Q_OBJECT
public:
    SkeletonJointEdit(QWidget *parent = 0);
    virtual ~SkeletonJointEdit();
protected:

private:
    QGroupBox * controlGrp;
    
public slots:
    void attachToJoint(int idx);
};
