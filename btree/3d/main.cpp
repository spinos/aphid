/*
 *  main.cpp
 *  
 *
 *  Created by jian zhang on 4/24/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include <BTree.h>
#include <Types.h>
#include <QApplication>
#include "window.h"
using namespace sdb;

int main(int argc, char *argv[])
{	
	QApplication app(argc, argv);
    Window window;
    //window.showMaximized();
	window.resize(800, 600);
	window.show();
    return app.exec();
}