/****************************************************************************
**
** Copyright (C) 2011 Nokia Corporation and/or its subsidiary(-ies).
** All rights reserved.
** Contact: Nokia Corporation (qt-info@nokia.com)
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** You may use this file under the terms of the BSD license as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
**     the names of its contributors may be used to endorse or promote
**     products derived from this software without specific prior written
**     permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
** $QT_END_LICENSE$
**
****************************************************************************/

#include <QtGui>

#include "treeitem.h"
#include "treemodel.h"
#include "delegate.h"
//! [0]
TreeModel::TreeModel(const QStringList &headers,
                     QObject *parent)
    : QAbstractItemModel(parent)
{
    QVector<QVariant> rootData;
    foreach (QString header, headers)
        rootData << header;

    rootItem = new TreeItem(rootData);
    setupModelData(rootItem);
}
//! [0]

//! [1]
TreeModel::~TreeModel()
{
    delete rootItem;
}
//! [1]

//! [2]
int TreeModel::columnCount(const QModelIndex & /* parent */) const
{
    return rootItem->columnCount();
}
//! [2]

QVariant TreeModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    if (role != Qt::DisplayRole && role != Qt::EditRole)
        return QVariant();

    TreeItem *item = getItem(index);

    return item->data(index.column());
}

//! [3]
Qt::ItemFlags TreeModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return 0;

	if(index.column() == 0)
		return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
		
	if(m_baseRows.find(getItem(index)->name()) != m_baseRows.end()) {
		return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
	}
    return Qt::ItemIsEditable | Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}
//! [3]

//! [4]
TreeItem *TreeModel::getItem(const QModelIndex &index) const
{
    if (index.isValid()) {
        TreeItem *item = static_cast<TreeItem*>(index.internalPointer());
        if (item) return item;
    }
    return rootItem;
}
//! [4]

QVariant TreeModel::headerData(int section, Qt::Orientation orientation,
                               int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
        return rootItem->data(section);

    return QVariant();
}

//! [5]
QModelIndex TreeModel::index(int row, int column, const QModelIndex &parent) const
{
    if (parent.isValid() && parent.column() != 0)
        return QModelIndex();
//! [5]

//! [6]
    TreeItem *parentItem = getItem(parent);

    TreeItem *childItem = parentItem->child(row);
    if (childItem)
        return createIndex(row, column, childItem);
    else
        return QModelIndex();
}
//! [6]

bool TreeModel::insertColumns(int position, int columns, const QModelIndex &parent)
{
    bool success;

    beginInsertColumns(parent, position, position + columns - 1);
    success = rootItem->insertColumns(position, columns);
    endInsertColumns();

    return success;
}

bool TreeModel::insertRows(int position, int rows, const QModelIndex &parent)
{
    TreeItem *parentItem = getItem(parent);
    bool success;

    beginInsertRows(parent, position, position + rows - 1);
    success = parentItem->insertChildren(position, rows, rootItem->columnCount());
    endInsertRows();

    return success;
}

//! [7]
QModelIndex TreeModel::parent(const QModelIndex &index) const
{
    if (!index.isValid())
        return QModelIndex();

    TreeItem *childItem = getItem(index);
    TreeItem *parentItem = childItem->parent();

    if (parentItem == rootItem)
        return QModelIndex();

    return createIndex(parentItem->childNumber(), 0, parentItem);
}
//! [7]

bool TreeModel::removeColumns(int position, int columns, const QModelIndex &parent)
{
    bool success;

    beginRemoveColumns(parent, position, position + columns - 1);
    success = rootItem->removeColumns(position, columns);
    endRemoveColumns();

    if (rootItem->columnCount() == 0)
        removeRows(0, rowCount());

    return success;
}

bool TreeModel::removeRows(int position, int rows, const QModelIndex &parent)
{
    TreeItem *parentItem = getItem(parent);
    bool success = true;

    beginRemoveRows(parent, position, position + rows - 1);
    success = parentItem->removeChildren(position, rows);
    endRemoveRows();

    return success;
}

//! [8]
int TreeModel::rowCount(const QModelIndex &parent) const
{
    TreeItem *parentItem = getItem(parent);

    return parentItem->childCount();
}
//! [8]

bool TreeModel::setData(const QModelIndex &index, const QVariant &value,
                        int role)
{
    if (role != Qt::EditRole)
        return false;

    TreeItem *item = getItem(index);
    bool result = item->setData(index.column(), value);

    if (result)
        emit dataChanged(index, index);

    return result;
}

bool TreeModel::setHeaderData(int section, Qt::Orientation orientation,
                              const QVariant &value, int role)
{
    if (role != Qt::EditRole || orientation != Qt::Horizontal)
        return false;

    bool result = rootItem->setData(section, value);

    if (result)
        emit headerDataChanged(orientation, section, section);

    return result;
}

void TreeModel::setupModelData(TreeItem *parent)
{
    QList<TreeItem*> parents;
    parents << parent;
	m_numRows = 0;
	addOptions(parents);
	addLights(parents);
	qDebug()<<"row count "<<m_numRows;
}

void TreeModel::addBase(QList<TreeItem*> & parents, const std::string & baseName, int level)
{
    for(int i=0; i < level; i++) 
        parents << parents.last()->child(parents.last()->childCount()-1);

    QList<QVariant> columnData;
    columnData << QString(tr(baseName.c_str()));
    
	TreeItem *parent = parents.last();
	parent->insertChildren(parent->childCount(), 1, rootItem->columnCount());
	for (int column = 0; column < columnData.size(); ++column)
		parent->child(parent->childCount() - 1)->setData(column, columnData[column]);
    
	for(int i=0; i < level; i++) parents.pop_back();
	m_baseRows[baseName] = 1;
	m_numRows++;
}

void TreeModel::addIntAttr(QList<TreeItem*> & parents, const std::string & attrName, int level, int value)
{
    for(int i=0; i < level; i++) 
        parents << parents.last()->child(parents.last()->childCount()-1);

    QList<QVariant> columnData;
    columnData << QString(tr(attrName.c_str()))<< QVariant(value);
    
	TreeItem *parent = parents.last();
	parent->insertChildren(parent->childCount(), 1, rootItem->columnCount());
	parent->lastChild()->setValueType(0);
	for (int column = 0; column < columnData.size(); ++column)
		parent->lastChild()->setData(column, columnData[column]);
		
    for(int i=0; i < level; i++) parents.pop_back();
	m_numRows++;
}

void TreeModel::addFltAttr(QList<TreeItem*> & parents, const std::string & attrName, int level, float value)
{
    for(int i=0; i < level; i++) 
        parents << parents.last()->child(parents.last()->childCount()-1);

    QList<QVariant> columnData;
    columnData << QString(tr(attrName.c_str()))<< QVariant(value);
    
	TreeItem *parent = parents.last();
	parent->insertChildren(parent->childCount(), 1, rootItem->columnCount());
	parent->lastChild()->setValueType(1);
	for (int column = 0; column < columnData.size(); ++column)
		parent->lastChild()->setData(column, columnData[column]);
		
    for(int i=0; i < level; i++) parents.pop_back();
	m_numRows++;
}

void TreeModel::addBolAttr(QList<TreeItem*> & parents, const std::string & attrName, int level, bool value)
{
    for(int i=0; i < level; i++) 
        parents << parents.last()->child(parents.last()->childCount()-1);

    QList<QVariant> columnData;
    columnData << QString(tr(attrName.c_str()))<< QVariant(value);
    
	TreeItem *parent = parents.last();
	parent->insertChildren(parent->childCount(), 1, rootItem->columnCount());
	parent->lastChild()->setValueType(2);
	for (int column = 0; column < columnData.size(); ++column)
		parent->lastChild()->setData(column, columnData[column]);
		
    for(int i=0; i < level; i++) parents.pop_back();
	m_numRows++;
}

void TreeModel::addRGBAttr(QList<TreeItem*> & parents, const std::string & attrName, int level, QColor value)
{
	for(int i=0; i < level; i++) 
        parents << parents.last()->child(parents.last()->childCount()-1);

    QList<QVariant> columnData;
    columnData << QString(tr(attrName.c_str()))<< QVariant(value);
    
	TreeItem *parent = parents.last();
	parent->insertChildren(parent->childCount(), 1, rootItem->columnCount());
	parent->lastChild()->setValueType(3);
	for (int column = 0; column < columnData.size(); ++column)
		parent->lastChild()->setData(column, columnData[column]);
		
    for(int i=0; i < level; i++) parents.pop_back();
	m_numRows++;
}

void TreeModel::addOptions(QList<TreeItem*> & parents)
{
	addBase(parents, "options", 0);
	addIntAttr(parents, "max_subdiv", 1, 3);
	addIntAttr(parents, "AA_samples", 1, 5);
	addIntAttr(parents, "res_x", 1, 400);
	addIntAttr(parents, "res_y", 1, 300);
}

void TreeModel::addLights(QList<TreeItem*> & parents)
{
	addBase(parents, "lights", 0);
	addBase(parents, "distant_key", 1);
	addFltAttr(parents, "intensity", 2, 1.1);
	addIntAttr(parents, "samples", 2, 3);
	addBolAttr(parents, "cast_shadow", 2, false);
	QColor col;
	col.setRgbF(1.0, 0.5, 0.4);
	addRGBAttr(parents, "light_color", 2, col);
}