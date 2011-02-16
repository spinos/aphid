#include <QtGui>
#include "changableButton.h"

ChangableButton::ChangableButton(QWidget *parent)
    : QWidget(parent)
{
    QLabel *addressLabel = new QLabel(tr("Note:"));
    
    togglePushButton = new QPushButton(tr("Click to Change Color"));
    togglePushButton->setCheckable(true);
    togglePushButton->setChecked(false);
    togglePushButton->setStyleSheet("QPushButton {background-color: darkkhaki; border-style: solid; border-radius: 5;} QPushButton:checked { background-color: green;}");
//! [constructor and input fields]

//! [layout]
    QGridLayout *mainLayout = new QGridLayout;
    mainLayout->addWidget(addressLabel, 0, 0);
    mainLayout->addWidget(togglePushButton, 0, 1);
//! [layout]

//![setting the layout]    
    setLayout(mainLayout);
    setWindowTitle(tr("Colorful Button"));
}

