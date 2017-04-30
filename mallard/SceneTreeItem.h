/*
 *  SceneTreeItem.h
 *  mallard
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
 
#pragma once

#include <QList>
#include <QVariant>
#include <QVector>

//! [0]
class SceneTreeItem
{
public:
	enum ValueType {
		TInt = 0,
		TFloat = 1,
		TBool = 2,
		TRGB = 3
	};
    SceneTreeItem(const QVector<QVariant> &data, SceneTreeItem *parent = 0);
    ~SceneTreeItem();

    SceneTreeItem *child(int number);
	SceneTreeItem * lastChild();
    int childCount() const;
    int columnCount() const;
    QVariant data(int column) const;
    bool insertChildren(int position, int count, int columns);
    bool insertColumns(int position, int columns);
    SceneTreeItem *parent();
	SceneTreeItem *parent() const;
    bool removeChildren(int position, int count);
    bool removeColumns(int position, int columns);
    int childNumber() const;
    bool setData(int column, const QVariant &value);
	QString name() const;
	QStringList fullPathName() const;
	std::string sname() const;
	void setValueType(int x);
	int valueType() const;
	void setLevel(int x);
private:
    QList<SceneTreeItem*> childItems;
    QVector<QVariant> itemData;
    SceneTreeItem *parentItem;
	int m_valueType;
	int m_level;
};
