/*
 *  SceneSceneTreeItem.h
 *  spinboxdelegate
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
 
#include <QtGui>
#include "SceneTreeModel.h"
#include "SceneTreeItem.h"
#include "MlScene.h"
#include "AllLight.h"
#include "AllEdit.h"

SceneTreeModel::SceneTreeModel(const QStringList &headers, 
                     QObject *parent)
    : QAbstractItemModel(parent)
{
	QVector<QVariant> rootData;
    foreach (QString header, headers)
        rootData << header;

    rootItem = new SceneTreeItem(rootData);
}
//! [0]

//! [1]
SceneTreeModel::~SceneTreeModel()
{
    delete rootItem;
}
//! [1]

//! [2]
int SceneTreeModel::columnCount(const QModelIndex & /* parent */) const
{
    return rootItem->columnCount();
}
//! [2]

QVariant SceneTreeModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    if (role != Qt::DisplayRole && role != Qt::EditRole)
        return QVariant();

    SceneTreeItem *item = getItem(index);

    return item->data(index.column());
}

//! [3]
Qt::ItemFlags SceneTreeModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return 0;

	if(index.column() == 0)
		return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
		
	if(m_baseRows.find(getItem(index)->sname()) != m_baseRows.end()) {
		return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
	}
    return Qt::ItemIsEditable | Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

SceneTreeItem * SceneTreeModel::getRootItem()
{
	return rootItem;
}

SceneTreeItem *SceneTreeModel::getItem(const QModelIndex &index) const
{
    if (index.isValid()) {
        SceneTreeItem *item = static_cast<SceneTreeItem*>(index.internalPointer());
        if (item) return item;
    }
    return rootItem;
}
//! [4]

QVariant SceneTreeModel::headerData(int section, Qt::Orientation orientation,
                               int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
        return rootItem->data(section);

    return QVariant();
}

//! [5]
QModelIndex SceneTreeModel::index(int row, int column, const QModelIndex &parent) const
{
    if (parent.isValid() && parent.column() != 0)
        return QModelIndex();
//! [5]

//! [6]
    SceneTreeItem *parentItem = getItem(parent);

    SceneTreeItem *childItem = parentItem->child(row);
    if (childItem)
        return createIndex(row, column, childItem);
    else
        return QModelIndex();
}
//! [6]

bool SceneTreeModel::insertColumns(int position, int columns, const QModelIndex &parent)
{
    bool success;

    beginInsertColumns(parent, position, position + columns - 1);
    success = rootItem->insertColumns(position, columns);
    endInsertColumns();

    return success;
}

bool SceneTreeModel::insertRows(int position, int rows, const QModelIndex &parent)
{
    SceneTreeItem *parentItem = getItem(parent);
    bool success;

    beginInsertRows(parent, position, position + rows - 1);
    success = parentItem->insertChildren(position, rows, rootItem->columnCount());
    endInsertRows();

    return success;
}

//! [7]
QModelIndex SceneTreeModel::parent(const QModelIndex &index) const
{
    if (!index.isValid())
        return QModelIndex();

    SceneTreeItem *childItem = getItem(index);
    SceneTreeItem *parentItem = childItem->parent();

    if (parentItem == rootItem)
        return QModelIndex();

    return createIndex(parentItem->childNumber(), 0, parentItem);
}
//! [7]

bool SceneTreeModel::removeColumns(int position, int columns, const QModelIndex &parent)
{
    bool success;

    beginRemoveColumns(parent, position, position + columns - 1);
    success = rootItem->removeColumns(position, columns);
    endRemoveColumns();

    if (rootItem->columnCount() == 0)
        removeRows(0, rowCount());

    return success;
}

bool SceneTreeModel::removeRows(int position, int rows, const QModelIndex &parent)
{
    SceneTreeItem *parentItem = getItem(parent);
    bool success = true;

    beginRemoveRows(parent, position, position + rows - 1);
    success = parentItem->removeChildren(position, rows);
    endRemoveRows();

    return success;
}

//! [8]
int SceneTreeModel::rowCount(const QModelIndex &parent) const
{
    SceneTreeItem *parentItem = getItem(parent);

    return parentItem->childCount();
}
//! [8]

bool SceneTreeModel::setData(const QModelIndex &index, const QVariant &value,
                        int role)
{
    if (role != Qt::EditRole)
        return false;

    SceneTreeItem *item = getItem(index);
    bool result = item->setData(index.column(), value);

    if (result)
        emit dataChanged(index, index);

    return result;
}

bool SceneTreeModel::setHeaderData(int section, Qt::Orientation orientation,
                              const QVariant &value, int role)
{
    if (role != Qt::EditRole || orientation != Qt::Horizontal)
        return false;

    bool result = rootItem->setData(section, value);

    if (result)
        emit headerDataChanged(orientation, section, section);

    return result;
}

void SceneTreeModel::addBase(QList<SceneTreeItem*> & parents, const std::string & baseName, int level)
{
    for(int i=0; i < level; i++) 
        parents << parents.last()->child(parents.last()->childCount()-1);

    QList<QVariant> columnData;
    columnData << QString(tr(baseName.c_str()));
    
	SceneTreeItem *parent = parents.last();
	parent->insertChildren(parent->childCount(), 1, rootItem->columnCount());
	parent->lastChild()->setLevel(level);
	for (int column = 0; column < columnData.size(); ++column)
		parent->child(parent->childCount() - 1)->setData(column, columnData[column]);
    
	for(int i=0; i < level; i++) parents.pop_back();
	m_baseRows[baseName] = 1;
}

void SceneTreeModel::addIntAttr(QList<SceneTreeItem*> & parents, const std::string & attrName, int level, int value)
{
    for(int i=0; i < level; i++) 
        parents << parents.last()->child(parents.last()->childCount()-1);

    QList<QVariant> columnData;
    columnData << QString(tr(attrName.c_str()))<< QVariant(value);
    
	SceneTreeItem *parent = parents.last();
	parent->insertChildren(parent->childCount(), 1, rootItem->columnCount());
	parent->lastChild()->setValueType(0);
	parent->lastChild()->setLevel(level);
	for (int column = 0; column < columnData.size(); ++column)
		parent->lastChild()->setData(column, columnData[column]);
		
    for(int i=0; i < level; i++) parents.pop_back();
}

void SceneTreeModel::addFltAttr(QList<SceneTreeItem*> & parents, const std::string & attrName, int level, float value)
{
    for(int i=0; i < level; i++) 
        parents << parents.last()->child(parents.last()->childCount()-1);

    QList<QVariant> columnData;
    columnData << QString(tr(attrName.c_str()))<< QVariant(value);
    
	SceneTreeItem *parent = parents.last();
	parent->insertChildren(parent->childCount(), 1, rootItem->columnCount());
	parent->lastChild()->setValueType(1);
	parent->lastChild()->setLevel(level);
	for (int column = 0; column < columnData.size(); ++column)
		parent->lastChild()->setData(column, columnData[column]);
		
    for(int i=0; i < level; i++) parents.pop_back();
}

void SceneTreeModel::addBolAttr(QList<SceneTreeItem*> & parents, const std::string & attrName, int level, bool value)
{
    for(int i=0; i < level; i++) 
        parents << parents.last()->child(parents.last()->childCount()-1);

    QList<QVariant> columnData;
    columnData << QString(tr(attrName.c_str()))<< QVariant(value);
    
	SceneTreeItem *parent = parents.last();
	parent->insertChildren(parent->childCount(), 1, rootItem->columnCount());
	parent->lastChild()->setValueType(2);
	parent->lastChild()->setLevel(level);
	for (int column = 0; column < columnData.size(); ++column)
		parent->lastChild()->setData(column, columnData[column]);
		
    for(int i=0; i < level; i++) parents.pop_back();
}

void SceneTreeModel::addRGBAttr(QList<SceneTreeItem*> & parents, const std::string & attrName, int level, QColor value)
{
	for(int i=0; i < level; i++) 
        parents << parents.last()->child(parents.last()->childCount()-1);

    QList<QVariant> columnData;
    columnData << QString(tr(attrName.c_str()))<< QVariant(value);
    
	SceneTreeItem *parent = parents.last();
	parent->insertChildren(parent->childCount(), 1, rootItem->columnCount());
	parent->lastChild()->setValueType(3);
	parent->lastChild()->setLevel(level);
	for (int column = 0; column < columnData.size(); ++column)
		parent->lastChild()->setData(column, columnData[column]);
		
    for(int i=0; i < level; i++) parents.pop_back();
}
