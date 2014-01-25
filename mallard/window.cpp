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
#include "BarbEdit.h"
#include "RenderEdit.h"
#include "SceneEdit.h"
Window::Window()
{
	std::cout<<"Initializing Mallard main window ";
    glWidget = new GLWidget;
	m_tools = new ToolBox;
	m_barbEdit = new BarbEdit(this);
	m_renderEdit = new RenderEdit(this);
	
	glWidget->setInteractContext(m_tools);
	m_brushControl = new BrushControl(glWidget->brush(), this);
	m_featherEdit = new FeatherEdit(this);
	FeatherExample::FeatherLibrary = glWidget;
	m_timeControl = new TimeControl(this);
	glWidget->setPlayback(m_timeControl);
	
	m_sceneEdit = new SceneEdit(glWidget, this);
	
	addToolBar(m_tools);

	setCentralWidget(glWidget);
    setWorkTitle(tr("untitled"));
	createActions();
	createMenus();
    
	connect(m_tools, SIGNAL(contextChanged(int)), this, SLOT(receiveToolContext(int)));
	connect(m_tools, SIGNAL(contextChanged(int)), m_brushControl, SLOT(receiveToolContext(int)));
    connect(m_tools, SIGNAL(actionTriggered(int)), this, SLOT(receiveToolAction(int)));
	connect(m_brushControl, SIGNAL(brushChanged()), glWidget, SLOT(receiveBrushChanged()));
	connect(glWidget, SIGNAL(sceneNameChanged(QString)), this, SLOT(setWorkTitle(QString)));
	connect(glWidget, SIGNAL(sendMessage(QString)), this, SLOT(showMessage(QString)));
	connect(m_featherEdit, SIGNAL(textureLoaded(QString)), glWidget, SLOT(receiveFeatherEditBackground(QString)));
	connect(m_featherEdit, SIGNAL(featherAdded()), glWidget, SLOT(receiveFeatherAdded()));
	connect(glWidget, SIGNAL(sendFeatherEditBackground(QString)), m_featherEdit, SLOT(receiveTexture(QString)));
	connect(m_timeControl, SIGNAL(currentFrameChanged(int)), glWidget, SLOT(updateOnFrame(int)));
	connect(m_featherEdit->uvView(), SIGNAL(selectionChanged()), m_barbEdit->barbControl(), SLOT(receiveSelectionChanged()));
	connect(glWidget, SIGNAL(featherSelectionChanged()), m_barbEdit->barbControl(), SLOT(receiveSelectionChanged()));
	connect(m_featherEdit->uvView(), SIGNAL(shapeChanged()), m_barbEdit->barbView(), SLOT(receiveShapeChanged()));
	connect(glWidget, SIGNAL(renderResChanged(QSize)), m_renderEdit, SLOT(resizeRenderView(QSize)));
	connect(glWidget, SIGNAL(renderEngineChanged(QString)), m_renderEdit, SLOT(setRenderEngine(QString)));
	connect(glWidget, SIGNAL(renderStarted(QString)), m_renderEdit, SLOT(startRender(QString)));
	connect(m_renderEdit, SIGNAL(cancelRender()), glWidget, SLOT(receiveCancelRender()));
	connect(m_barbEdit->barbControl(), SIGNAL(shapeChanged()), glWidget, SLOT(receiveBarbChanged()));
	connect(glWidget, SIGNAL(sceneOpened()), m_sceneEdit, SLOT(reloadScene()));
	connect(m_sceneEdit->model(), SIGNAL(cameraChanged()), glWidget, SLOT(receiveCameraChanged()));
	
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
		case ToolContext::LaunchRender:
			m_barbEdit->close();
		    m_renderEdit->show();
			glWidget->testRender();
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
	
    showBarbEditAct = new QAction(tr("&Barb Preview"), this);
    showBarbEditAct->setStatusTip(tr("Show barb preview"));
    connect(showBarbEditAct, SIGNAL(triggered()), m_barbEdit, SLOT(show()));
	
	showRenderEditAct = new QAction(tr("&Render View"), this);
    showRenderEditAct->setStatusTip(tr("Show render view"));
    connect(showRenderEditAct, SIGNAL(triggered()), m_renderEdit, SLOT(show()));
	
	showSceneEditAct = new QAction(tr("&Scene Tree"), this);
    showSceneEditAct->setStatusTip(tr("Show scene edit"));
    connect(showSceneEditAct, SIGNAL(triggered()), m_sceneEdit, SLOT(show()));
    
	saveSceneAct = new QAction(tr("&Save"), this);
	saveSceneAct->setStatusTip(tr("Save current scene file"));
    connect(saveSceneAct, SIGNAL(triggered()), glWidget, SLOT(saveSheet()));
	
	saveSceneAsAct = new QAction(tr("&Save As"), this);
	saveSceneAsAct->setStatusTip(tr("Save current scene into another file"));
	connect(saveSceneAsAct, SIGNAL(triggered()), glWidget, SLOT(saveSheetAs()));
	
	importMeshAct = new QAction(tr("&Import Body Mesh"), this);
	importMeshAct->setStatusTip(tr("Load a mesh cache file as the body"));
	connect(importMeshAct, SIGNAL(triggered()), glWidget, SLOT(importBodyMesh()));
	
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
	
	growOnAct = new QAction(tr("Eable Grow On Faces"), this);
	growOnAct->setStatusTip(tr("Tag selected faces to grow feathers"));
	connect(growOnAct, SIGNAL(triggered()), this, SLOT(tagFaceOn()));
	
	growOffAct = new QAction(tr("Disable Grow On Faces"), this);
	growOffAct->setStatusTip(tr("Tag selected faces NOT to grow feathers"));
    connect(growOffAct, SIGNAL(triggered()), this, SLOT(tagFaceOff()));
	
	displayFeatherOnAct = new QAction(tr("&Show Feather"), this);
	displayFeatherOnAct->setStatusTip(tr("Turn on feather display"));
	connect(displayFeatherOnAct, SIGNAL(triggered()), this, SLOT(displayFeatherOn()));
	
	displayFeatherOffAct = new QAction(tr("&Hide Feather"), this);
	displayFeatherOffAct->setStatusTip(tr("Turn off feather display"));
	connect(displayFeatherOffAct, SIGNAL(triggered()), this, SLOT(displayFeatherOff()));
	
	selectDistributeAct = new QAction(tr("&Grow Distribution"), this);
	selectDistributeAct->setStatusTip(tr("Choose to activate distribution map"));
	connect(selectDistributeAct, SIGNAL(triggered()), this, SLOT(selectMap1()));
	
	selectDensityAct = new QAction(tr("&Grow Density"), this);
	selectDensityAct->setStatusTip(tr("Choose to activate density map"));
	connect(selectDensityAct, SIGNAL(triggered()), this, SLOT(selectMap2()));
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
	windowMenu->addAction(showBarbEditAct);
	windowMenu->addAction(showBrushControlAct);
	windowMenu->addAction(showSceneEditAct);
	windowMenu->addAction(showTimeControlAct);
	windowMenu->addAction(showRenderEditAct);
	
	skinMenu = menuBar()->addMenu(tr("&Skin"));
	skinMenu->addAction(growOnAct);
	skinMenu->addAction(growOffAct);
	skinMenu->addAction(displayFeatherOnAct);
	skinMenu->addAction(displayFeatherOffAct);
	
	selectMapMenu = skinMenu->addMenu(tr("&Select Map"));
	selectMapMenu->addAction(selectDistributeAct);
	selectMapMenu->addAction(selectDensityAct);
	
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
	glWidget->receiveCancelRender();
	((MlScene *)glWidget)->close();
	QMainWindow::closeEvent(event);
}

void Window::tagFaceOn() { glWidget->tagFace(true); }
void Window::tagFaceOff() { glWidget->tagFace(false); }

void Window::displayFeatherOn() { glWidget->setDisplayFeather(true); }
void Window::displayFeatherOff() { glWidget->setDisplayFeather(false); }

void Window::selectMap1() { glWidget->selectMap(1); }
void Window::selectMap2() { glWidget->selectMap(2); }