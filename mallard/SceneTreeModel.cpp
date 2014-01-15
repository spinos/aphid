#include <QtGui>
#include "MlScene.h"
#include "SceneTreeItem.h"
#include "SceneTreeModel.h"

//! [0]
SceneTreeModel::SceneTreeModel(MlScene * scene, QObject *parent)
    : QAbstractItemModel(parent)
{
    QList<QVariant> rootData;
    rootData << "Attribute" << "Value";
    rootItem = new SceneTreeItem(rootData);
    setupModelData(scene, rootItem);
}
//! [0]

//! [1]
SceneTreeModel::~SceneTreeModel()
{
    delete rootItem;
}
//! [1]

//! [2]
int SceneTreeModel::columnCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return static_cast<SceneTreeItem*>(parent.internalPointer())->columnCount();
    else
        return rootItem->columnCount();
}
//! [2]

//! [3]
QVariant SceneTreeModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    if (role != Qt::DisplayRole)
        return QVariant();

    SceneTreeItem *item = static_cast<SceneTreeItem*>(index.internalPointer());

    return item->data(index.column());
}
//! [3]

//! [4]
Qt::ItemFlags SceneTreeModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return 0;

    return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}
//! [4]

//! [5]
QVariant SceneTreeModel::headerData(int section, Qt::Orientation orientation,
                               int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
        return rootItem->data(section);

    return QVariant();
}
//! [5]

//! [6]
QModelIndex SceneTreeModel::index(int row, int column, const QModelIndex &parent)
            const
{
    if (!hasIndex(row, column, parent))
        return QModelIndex();

    SceneTreeItem *parentItem;

    if (!parent.isValid())
        parentItem = rootItem;
    else
        parentItem = static_cast<SceneTreeItem*>(parent.internalPointer());

    SceneTreeItem *childItem = parentItem->child(row);
    if (childItem)
        return createIndex(row, column, childItem);
    else
        return QModelIndex();
}
//! [6]

//! [7]
QModelIndex SceneTreeModel::parent(const QModelIndex &index) const
{
    if (!index.isValid())
        return QModelIndex();

    SceneTreeItem *childItem = static_cast<SceneTreeItem*>(index.internalPointer());
    SceneTreeItem *parentItem = childItem->parent();

    if (parentItem == rootItem)
        return QModelIndex();

    return createIndex(parentItem->row(), 0, parentItem);
}
//! [7]

//! [8]
int SceneTreeModel::rowCount(const QModelIndex &parent) const
{
    SceneTreeItem *parentItem;
    if (parent.column() > 0)
        return 0;

    if (!parent.isValid())
        parentItem = rootItem;
    else
        parentItem = static_cast<SceneTreeItem*>(parent.internalPointer());

    return parentItem->childCount();
}
//! [8]

void SceneTreeModel::setupModelData(MlScene * scene, SceneTreeItem *parent)
{
    QList<SceneTreeItem*> parents;
    QList<int> indentations;
    parents << parent;
    indentations << 0;

    int number = 0;
/*
    while (number < lines.count()) {
        int position = 0;
        while (position < lines[number].length()) {
            if (lines[number].mid(position, 1) != " ")
                break;
            position++;
        }

        QString lineData = lines[number].mid(position).trimmed();

        if (!lineData.isEmpty()) {
            // Read the column data from the rest of the line.
            QStringList columnStrings = lineData.split("\t", QString::SkipEmptyParts);
            QList<QVariant> columnData;
            for (int column = 0; column < columnStrings.count(); ++column)
                columnData << columnStrings[column];

            if (position > indentations.last()) {
                // The last child of the current parent is now the new parent
                // unless the current parent has no children.

                if (parents.last()->childCount() > 0) {
                    parents << parents.last()->child(parents.last()->childCount()-1);
                    indentations << position;
                }
            } else {
                while (position < indentations.last() && parents.count() > 0) {
                    parents.pop_back();
                    indentations.pop_back();
                }
            }

            // Append a new item to the current parent's list of children.
            parents.last()->appendChild(new SceneTreeItem(columnData, parents.last()));
        }

        number++;
    }*/
    addBase(parents, "options", 0);
    addBase(parents, "lights", 0);
    addBase(parents, "key", 1);
    addBase(parents, "kd", 2);
    addBase(parents, "models", 0);
    addBase(parents, "body", 1);
    addBase(parents, "shaders", 0);
    addBase(parents, "hair", 1);
}

void SceneTreeModel::addBase(QList<SceneTreeItem*> & parents, const std::string & baseName, int level)
{
    for(int i=0; i < level; i++) 
        parents << parents.last()->child(parents.last()->childCount()-1);

    QList<QVariant> columnData;
    columnData << QString(tr(baseName.c_str())) << QString(tr("1"));
    parents.last()->appendChild(new SceneTreeItem(columnData, parents.last()));
    for(int i=0; i < level; i++) parents.pop_back();
    qDebug()<<parents;
    
}
