/*
 *  SceneTreeItem.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include <QStringList>

#include "SceneTreeItem.h"

//! [0]
SceneTreeItem::SceneTreeItem(const QVector<QVariant> &data, SceneTreeItem *parent)
{
    parentItem = parent;
    itemData = data;
}
//! [0]

//! [1]
SceneTreeItem::~SceneTreeItem()
{
    qDeleteAll(childItems);
}
//! [1]

//! [2]
SceneTreeItem *SceneTreeItem::child(int number)
{
    return childItems.value(number);
}

SceneTreeItem * SceneTreeItem::lastChild()
{
	if(childCount() < 1) return 0;
	return child(childCount() - 1);
}

int SceneTreeItem::childCount() const
{
    return childItems.count();
}
//! [3]

//! [4]
int SceneTreeItem::childNumber() const
{
    if (parentItem)
        return parentItem->childItems.indexOf(const_cast<SceneTreeItem*>(this));

    return 0;
}
//! [4]

//! [5]
int SceneTreeItem::columnCount() const
{
    return itemData.count();
}
//! [5]

//! [6]
QVariant SceneTreeItem::data(int column) const
{
    return itemData.value(column);
}
//! [6]

//! [7]
bool SceneTreeItem::insertChildren(int position, int count, int columns)
{
    if (position < 0 || position > childItems.size())
        return false;

    for (int row = 0; row < count; ++row) {
        QVector<QVariant> data(columns);
        SceneTreeItem *item = new SceneTreeItem(data, this);
        childItems.insert(position, item);
    }

    return true;
}
//! [7]

//! [8]
bool SceneTreeItem::insertColumns(int position, int columns)
{
    if (position < 0 || position > itemData.size())
        return false;

    for (int column = 0; column < columns; ++column)
        itemData.insert(position, QVariant());

    foreach (SceneTreeItem *child, childItems)
        child->insertColumns(position, columns);

    return true;
}
//! [8]

//! [9]
SceneTreeItem *SceneTreeItem::parent()
{
    return parentItem;
}
//! [9]

//! [10]
bool SceneTreeItem::removeChildren(int position, int count)
{
    if (position < 0 || position + count > childItems.size())
        return false;

    for (int row = 0; row < count; ++row)
        delete childItems.takeAt(position);

    return true;
}
//! [10]

bool SceneTreeItem::removeColumns(int position, int columns)
{
    if (position < 0 || position + columns > itemData.size())
        return false;

    for (int column = 0; column < columns; ++column)
        itemData.remove(position);

    foreach (SceneTreeItem *child, childItems)
        child->removeColumns(position, columns);

    return true;
}

//! [11]
bool SceneTreeItem::setData(int column, const QVariant &value)
{
    if (column < 0 || column >= itemData.size())
        return false;

    itemData[column] = value;
    return true;
}
//! [11]
std::string SceneTreeItem::name() const
{
	return data(0).toString().toStdString();
}

void SceneTreeItem::setValueType(int x)
{
	m_valueType = x;
}

int SceneTreeItem::valueType() const
{
	return m_valueType;
}
