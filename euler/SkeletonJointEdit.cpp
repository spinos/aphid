#include <QtGui>
#include "SkeletonJointEdit.h"
#include <sstream>

SkeletonJointEdit::SkeletonJointEdit(QWidget *parent) : QWidget(parent)
{
    controlGrp = new QGroupBox(tr("unknown"));
    
    QVBoxLayout * main = new QVBoxLayout;
    main->addWidget(controlGrp);
    setLayout(main);
}

SkeletonJointEdit::~SkeletonJointEdit() {}

void SkeletonJointEdit::attachToJoint(int idx)
{
    std::stringstream sst;
    sst.str("");
    sst<<idx;
    controlGrp->setTitle(tr(sst.str().c_str()));
}
