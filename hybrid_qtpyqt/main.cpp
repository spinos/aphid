/****************************************************************************
**
** Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
** All rights reserved.
** Contact: Nokia Corporation (qt-info@nokia.com)
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:LGPL$
** Commercial Usage
** Licensees holding valid Qt Commercial licenses may use this file in
** accordance with the Qt Commercial License Agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and Nokia.
**
** GNU Lesser General Public License Usage
** Alternatively, this file may be used under the terms of the GNU Lesser
** General Public License version 2.1 as published by the Free Software
** Foundation and appearing in the file LICENSE.LGPL included in the
** packaging of this file.  Please review the following information to
** ensure the GNU Lesser General Public License version 2.1 requirements
** will be met: http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
**
** In addition, as a special exception, Nokia gives you certain additional
** rights.  These rights are described in the Nokia Qt LGPL Exception
** version 1.1, included in the file LGPL_EXCEPTION.txt in this package.
**
** GNU General Public License Usage
** Alternatively, this file may be used under the terms of the GNU
** General Public License version 3.0 as published by the Free Software
** Foundation and appearing in the file LICENSE.GPL included in the
** packaging of this file.  Please review the following information to
** ensure the GNU General Public License version 3.0 requirements will be
** met: http://www.gnu.org/copyleft/gpl.html.
**
** If you have questions regarding the use of this file, please contact
** Nokia at qt-info@nokia.com.
** $QT_END_LICENSE$
**
****************************************************************************/

#include <QtGui>
#include <QMainWindow>

#include <python.h>
#include <../hoatzin/hoatzin.h>

#include <iostream>
using namespace std;

class LeftWidget : public HWidget
{
public:
	LeftWidget();
	void setAttribute(const char *attributeName, int value);
private:
	QLineEdit *attributeEdit;
	QLineEdit *attribute1Edit;
};

LeftWidget::LeftWidget()
{
	setObjectName("SnakeHole");
	QHBoxLayout* box = new QHBoxLayout();
	box->addWidget(new QLabel("qt host"));
	attributeEdit = new QLineEdit();
    box->addWidget(attributeEdit);
	attribute1Edit = new QLineEdit();
    box->addWidget(attribute1Edit);
	setLayout(box);
	
}

void LeftWidget::setAttribute(const char *attributeName, int value)
{
	QString attr(attributeName);
	if(attr == "fieldNumber")
	{
		QString str = QString("%1")
             .arg(value);
		attributeEdit->setText(str);
	}
	else if(attr == "sliderNumber")
	{
		QString str = QString("%1")
             .arg(value);
		attribute1Edit->setText(str);
	}
}


class ShakeHole : public QWidget
{
public:
    ShakeHole();
	~ShakeHole() {}
    
};


ShakeHole::ShakeHole()
{
	
    int wnd = winId();
        cout<<"winid "<<wnd<<endl;
        
        
    cout<<"version "<<Py_GetVersion()<<endl;
	
	FILE *fin = fopen("/Users/jianzhang/aphid/hybrid_qtpyqt/foo.py","r+");
	PyRun_SimpleFile(fin,"foo");
	
        PyObject *mainDict = PyModule_GetDict(PyImport_Import(PyString_FromString("__main__")));
        
        PyObject *claus = PyDict_GetItemString(mainDict, "Ui_Form");
        
        PyObject* snakewig = Py_BuildValue("i", wnd);
        
        PyObject * pTuple = PyTuple_New (1);
        PyTuple_SetItem (pTuple, 0, snakewig);
        
        PyObject *clausInstance = PyObject_CallObject(claus, pTuple);
		
		cout<<"finished python ui!\n";
        
		PyErr_Print();
	
}



class Center : public QWidget
{
public:
    Center();   
};

Center::Center()
{
	setObjectName("center");
    QVBoxLayout* box = new QVBoxLayout();
    LeftWidget *leftDial = new LeftWidget();
    box->addWidget(leftDial);
    ShakeHole *rightDial = new ShakeHole();
    box->addWidget(rightDial);
    setLayout(box);
	
	//connect(rightDial, SIGNAL(intvalue(int)), leftDial, SLOT(setValue(int)));
}

class Window : public QMainWindow
{
public:
    Window();
};

Window::Window()
{
    Center* center = new Center();
    setCentralWidget(center);
    setWindowTitle(tr("Qt PyQt Hybrid"));
}

//! [main function]
int main(int argc, char *argv[])
{
	Py_Initialize();
    QApplication app(argc, argv);
	
	

	Window window;
    window.show();

    return app.exec();
	
    Py_Finalize();
}
//! [main function]
