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

#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <SingleModelView.h>
#include <MlScene.h>

class BezierDrawer;
class MlDrawer;

//! [0]
class GLWidget : public SingleModelView, public MlScene
{
    Q_OBJECT

public:
    GLWidget(QWidget *parent = 0);
    ~GLWidget();
	virtual void setFeatherTexture(const std::string & name);
	virtual void loadMesh(std::string filename);
	virtual void clientSelect();
	virtual void clientMouseInput();
	virtual void clientDeselect();
    virtual PatchMesh * mesh();
	
	virtual bool confirmDiscardChanges();
	virtual std::string chooseOpenFileName();
	virtual std::string chooseSaveFileName();
	virtual void doClear();
	virtual void doClose();
	virtual void beforeSave();
	virtual void afterOpen();
	
	void finishEraseFeather();
	void deselectFeather();
	void rebuildFeather();
	void clearFeather();
	void bakeFrames();
	
	QString openSheet(QString name);
	
signals:
	void sceneNameChanged(QString name);
	void sendMessage(QString msg);
	void sendFeatherEditBackground(QString name);
	
public slots:
	void cleanSheet();
	void saveSheet();
	void saveSheetAs();
    void revertSheet();
	void receiveFeatherEditBackground(QString name);
	void chooseBake();
	void updateOnFrame(int x);
	void exportBake();
	void importFeatherDistributionMap();
	void receiveFloodRegion(int state);
protected:
    virtual void clientDraw();
	virtual void focusOutEvent(QFocusEvent * event);
	virtual void clearSelection();
	
//! [3]
private:
	void selectFeather();
	void selectRegion();
	void floodFeather();
	void beginBaking();
	void endBaking();
	bool isBaking() const;
	void loadFeatherDistribution(const std::string & name);
private:
	BezierDrawer * m_bezierDrawer;
	MlDrawer * m_featherDrawer;
	int m_featherTexId, m_featherDistrId;
};
//! [3]

#endif
