/*
 *  ContextToolGroup.h
 *  
 *  tool box with multiple icons
 *  one tool at a time
 *
 *  Created by jian zhang on 9/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_CONTEXT_TOOL_GROUP_H
#define APH_CONTEXT_TOOL_GROUP_H

#include <QGroupBox>

QT_BEGIN_NAMESPACE
class QPixmap;
QT_END_NAMESPACE

namespace aphid {

class QIconFrame;

class ContextToolProfile {

public:
	struct AToolContext {
		int _name;
		QString _icon[2];
	};
	
private:
	QString m_title;
	QList<AToolContext > m_ctx;
	
public:
	ContextToolProfile();
	~ContextToolProfile();
	
	void setTitle(const QString& x);
	void addTool(int toolName, const QString& iconName0,
					const QString& iconName1);
	
	const QString& title() const;
	int numTools() const;
	const AToolContext& getTool(int& row, int& col,
				const int& i) const;
	int numCols() const;

private:
};

class ContextToolGroup : public QGroupBox 
{
	Q_OBJECT
public:
	ContextToolGroup(ContextToolProfile* prof,
			QWidget *parent = 0);
	
	void setNameId(int x);
	const int& nameId() const;
	
signals:
	void toolSelected2(QPair<int, int> x);
	
private slots:
	void recvContextChange(QPair<int, int> x);
	
private:
	QList<QIconFrame* > m_tools;
	int m_nameId;
	
};

}
#endif