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

#include "glwidget.h"
#include "ToolBox.h"
#include "BrushControl.h"
#include "FeatherEdit.h"
#include "MlUVView.h"
#include "window.h"
#include "TimeControl.h"
//! [0]
Window::Window()
{
	std::cout<<"Initializing Mallard main window ";
    glWidget = new GLWidget;
	m_tools = new ToolBox;
	
	glWidget->setInteractContext(m_tools);
	m_brushControl = new BrushControl(this);
	m_featherEdit = new FeatherEdit(this);
	MlUVView::FeatherLibrary = glWidget;
	m_timeControl = new TimeControl(this);
	glWidget->setPlayback(m_timeControl);
	addToolBar(m_tools);

	setCentralWidget(glWidget);
    setWorkTitle(tr("untitled"));
	createActions();
	createMenus();
    
	connect(m_tools, SIGNAL(contextChanged(int)), this, SLOT(receiveToolContext(int)));
	connect(m_tools, SIGNAL(contextChanged(int)), m_brushControl, SLOT(receiveToolContext(int)));
    connect(m_tools, SIGNAL(actionTriggered(int)), this, SLOT(receiveToolAction(int)));
	connect(m_brushControl->pitchWidget(), SIGNAL(valueChanged(double)), glWidget, SLOT(receiveBrushPitch(double)));
	connect(m_brushControl, SIGNAL(brushRadiusChanged(double)), glWidget, SLOT(receiveBrushRadius(double)));
	connect(m_brushControl->numSamplesWidget(), SIGNAL(valueChanged(int)), glWidget, SLOT(receiveBrushNumSamples(int)));
	connect(m_brushControl->floodRegionWidget(), SIGNAL(stateChanged(int)), glWidget, SLOT(receiveFloodRegion(int)));
	connect(m_brushControl->eraseRegionWidget(), SIGNAL(stateChanged(int)), glWidget, SLOT(receiveEraseRegion(int)));
	connect(m_brushControl->eraseStrengthWidget(), SIGNAL(valueChanged(double)), glWidget, SLOT(receiveBrushStrength(double)));
	connect(glWidget, SIGNAL(sceneNameChanged(QString)), this, SLOT(setWorkTitle(QString)));
	connect(glWidget, SIGNAL(sendMessage(QString)), this, SLOT(showMessage(QString)));
	connect(m_featherEdit, SIGNAL(textureLoaded(QString)), glWidget, SLOT(receiveFeatherEditBackground(QString)));
	connect(glWidget, SIGNAL(sendFeatherEditBackground(QString)), m_featherEdit, SLOT(receiveTexture(QString)));
	connect(m_timeControl, SIGNAL(currentFrameChanged(int)), glWidget, SLOT(updateOnFrame(int)));
	
	std::cout<<"Ready\n";
	statusBar()->showMessage(tr("Ready"));
}

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}

void Window::receiveToolContext(int a)
{
	if(m_tools->previousContext() == ToolContext::EraseBodyContourFeather)
		glWidget->finishEraseFeather();
	glWidget->deselectFeather();
}

void Window::receiveToolAction(int a)
{
	switch (a) {
		case ToolContext::RebuildBodyContourFeather:
			glWidget->rebuildFeather();
			break;
		case ToolContext::ClearBodyContourFeather:
			glWidget->clearFeather();
			break;
		case ToolContext::BakeAnimation:
			glWidget->bakeFrames();
			break;
		default:
			break;
	}
}

void Window::createActions()
{
    newSceneAct = new QAction(tr("&New"), this);
    newSceneAct->setStatusTip(tr("Create an empty scene"));
    connect(newSceneAct, SIGNAL(triggered()), glWidget, SLOT(cleanSheet()));
	
	showBrushControlAct = new QAction(tr("&Brush Control"), this);
	showBrushControlAct->setStatusTip(tr("Show brush settings"));
    connect(showBrushControlAct, SIGNAL(triggered()), m_brushControl, SLOT(show()));
	
	showFeatherEditAct = new QAction(tr("&Feather Editor"), this);
	showFeatherEditAct->setStatusTip(tr("Show feather editor"));
    connect(showFeatherEditAct, SIGNAL(triggered()), m_featherEdit, SLOT(show()));
	
	showTimeControlAct = new QAction(tr("&Time Control"), this);
	showTimeControlAct->setStatusTip(tr("Show time control"));
    connect(showTimeControlAct, SIGNAL(triggered()), m_timeControl, SLOT(show()));
	
	saveSceneAct = new QAction(tr("&Save"), this);
	saveSceneAct->setStatusTip(tr("Save current scene file"));
    connect(saveSceneAct, SIGNAL(triggered()), glWidget, SLOT(saveSheet()));
	
	saveSceneAsAct = new QAction(tr("&Save As"), this);
	saveSceneAsAct->setStatusTip(tr("Save current scene into another file"));
	connect(saveSceneAsAct, SIGNAL(triggered()), glWidget, SLOT(saveSheetAs()));
	
	importMeshAct = new QAction(tr("&Import Body Mesh"), this);
	importMeshAct->setStatusTip(tr("Load a mesh cache file as the body"));
	connect(importMeshAct, SIGNAL(triggered()), glWidget, SLOT(open()));
	
	importFDMAct = new QAction(tr("&Import Feather Distribution Map"), this);
	importFDMAct->setStatusTip(tr("Load an OpenEXR image as feather distribution"));
	connect(importFDMAct, SIGNAL(triggered()), glWidget, SLOT(importFeatherDistributionMap()));
	
	openSceneAct = new QAction(tr("&Open"), this);
	openSceneAct->setStatusTip(tr("Load a file as current scene"));
	connect(openSceneAct, SIGNAL(triggered()), this, SLOT(openFile()));
	
	revertAct = new QAction(tr("&Revert To Saved"), this);
	revertAct->setStatusTip(tr("Discard changes to current scene after lastest save"));
    connect(revertAct, SIGNAL(triggered()), glWidget, SLOT(revertSheet()));
	
	importBakeAct = new QAction(tr("&Import Body Animation"), this);
	importBakeAct->setStatusTip(tr("Attach bake point cache to the body"));
    connect(importBakeAct, SIGNAL(triggered()), glWidget, SLOT(chooseBake()));
	
	for (int i = 0; i < MaxRecentFiles; ++i) {
        recentFileActs[i] = new QAction(this);
        recentFileActs[i]->setVisible(false);
        connect(recentFileActs[i], SIGNAL(triggered()),
                this, SLOT(openRecentFile()));
    }
    
    exportBakeAct = new QAction(tr("&Export Baked Feather"), this);
	exportBakeAct->setStatusTip(tr("Write feather cache"));
    connect(exportBakeAct, SIGNAL(triggered()), glWidget, SLOT(exportBake()));
}

void Window::createMenus()
{
    fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(newSceneAct);
	fileMenu->addAction(openSceneAct);
	fileMenu->addAction(saveSceneAct);
	fileMenu->addAction(saveSceneAsAct);
	fileMenu->addAction(revertAct);
	
	recentFilesMenu = new QMenu(tr("&Recent Scene"));
	for (int i = 0; i < MaxRecentFiles; ++i)
        recentFilesMenu->addAction(recentFileActs[i]);
	fileMenu->addMenu(recentFilesMenu);
	
	fileMenu->addSeparator();
	fileMenu->addAction(importMeshAct);
	fileMenu->addAction(importFDMAct);
	fileMenu->addAction(importBakeAct);
	fileMenu->addAction(exportBakeAct);
	
	windowMenu = menuBar()->addMenu(tr("&Window"));
    windowMenu->addAction(showFeatherEditAct);
	windowMenu->addAction(showBrushControlAct);
	windowMenu->addAction(showTimeControlAct);
	
	updateRecentFileActions();
}

void Window::setWorkTitle(QString name)
{
	setWindowTitle(QString("Mallard - %1").arg(name));
}

void Window::showMessage(QString msg)
{
	statusBar()->showMessage(msg);
}

void Window::openFile()
{
	QString fileName = glWidget->openSheet(tr(""));
	if(fileName == "") return;
	
	QSettings settings("mallard.ini", QSettings::IniFormat, this);

    QStringList files = settings.value("recentFileList").toStringList();
    files.removeAll(fileName);
    files.prepend(fileName);
    while (files.size() > MaxRecentFiles)
        files.removeLast();

    settings.setValue("recentFileList", files);

    updateRecentFileActions();
}
	
void Window::openRecentFile()
{
	QAction *action = qobject_cast<QAction *>(sender());
    if (action)
		glWidget->openSheet(action->data().toString());
}

void Window::updateRecentFileActions()
{
    QSettings settings("mallard.ini", QSettings::IniFormat, this);
	
    QStringList files = settings.value("recentFileList").toStringList();

    int numRecentFiles = qMin(files.size(), (int)MaxRecentFiles);

    for (int i = 0; i < numRecentFiles; ++i) {
        QString text = tr("&%1").arg(files[i]);
        recentFileActs[i]->setText(text);
        recentFileActs[i]->setData(files[i]);
        recentFileActs[i]->setVisible(true);
    }
    for (int j = numRecentFiles; j < MaxRecentFiles; ++j)
        recentFileActs[j]->setVisible(false);
}

void Window::closeEvent(QCloseEvent *event)
{
	if(!((MlScene *)glWidget)->close()) event->ignore();
}