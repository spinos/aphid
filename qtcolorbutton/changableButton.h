#ifndef CHANGABLEBUTTON_H
#define CHANGABLEBUTTON_H

#include <QWidget>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QTextEdit;
class QPushButton;
QT_END_NAMESPACE

//! [class definition]
class ChangableButton : public QWidget
{
    Q_OBJECT

public:
    ChangableButton(QWidget *parent = 0);

private:
    QPushButton *togglePushButton;
};

#endif        //  #ifndef CHANGABLEBUTTON_H

