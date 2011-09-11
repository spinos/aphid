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
#include "addressbook.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <compact_implement.h>
//! [constructor and input fields]
AddressBook::AddressBook(QWidget *parent)
    : QWidget(parent)
{
	checkCUDevice();
	
	QImage *imagefile = new QImage(":images/eagle.jpg");
	qDebug() << "image depth: " << imagefile->depth();
	QSize imagesize = imagefile->size();
	qDebug() << QString("image size: %1 x %2")
                     .arg(imagesize.width()).arg(imagesize.height());
	
        //unsigned char *luminancebits = new unsigned char[imagesize.width()*imagesize.height()*4];				 
	//initTexture(imagesize.width(), imagesize.height(), imagefile->bits(), luminancebits); 
	
	unsigned char *countbits = new unsigned char[imagesize.width()*imagesize.height()*4];
	countTexture(imagesize.width(), imagesize.height(), imagefile->bits(), countbits);

	inimagelabel = new QLabel();
	inimagelabel->setPixmap(QPixmap::fromImage(*imagefile));
	
	//QImage *luminanceImage = new QImage(luminancebits, imagesize.width(), imagesize.height(), QImage::Format_ARGB32);
	
	//QPixmap outimage = QPixmap::fromImage(*luminanceImage);
	//outimagelabel = new QLabel();
	//outimagelabel->setPixmap(outimage);
	
	QLabel *compactimagelabel = new QLabel();
	QImage *compactImage = new QImage(countbits, imagesize.width(), imagesize.height(), QImage::Format_ARGB32);
	
	QPixmap compactpix = QPixmap::fromImage(*compactImage);
	compactimagelabel->setPixmap(compactpix);
//! [constructor and input fields]

//! [layout]
    QGridLayout *mainLayout = new QGridLayout;

	mainLayout->addWidget(inimagelabel, 0, 0);
	
	mainLayout->addWidget(compactimagelabel, 0, 1);
	//mainLayout->addWidget(outimagelabel, 1, 1);
//! [layout]

//![setting the layout]    
    setLayout(mainLayout);
    setWindowTitle(tr("Compaction"));
}
//! [setting the layout]

void AddressBook::checkCUDevice()
{
int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
	{
		qDebug() << "Cannot find CUDA device!";
	}
	if(deviceCount>0)
	{
		qDebug() << "Found " << deviceCount << " device(s)";
		int driverVersion = 0, runtimeVersion = 0;
		cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		qDebug() << "Device name: " << deviceProp.name;
		qDebug() << "  Diver Version: " << driverVersion;
		qDebug() << "  Runtime Version: " << runtimeVersion;
		
	qDebug() << QString("  Maximum sizes of each dimension of a grid: %1 x %2 x %3")
               .arg(deviceProp.maxGridSize[0]).arg(deviceProp.maxGridSize[1]).arg(deviceProp.maxGridSize[2]);
	
        qDebug() << QString("  Maximum sizes of each dimension of a block: %1 x %2 x %3")
                     .arg(deviceProp.maxThreadsDim[0]).arg(deviceProp.maxThreadsDim[1]).arg(deviceProp.maxThreadsDim[2]);
                qDebug() << "  Maximum number of threads per block: " << deviceProp.maxThreadsPerBlock;
        		   
	}
}
