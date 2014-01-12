#include <QStringList>

#include "SceneTreeItem.h"

//! [0]
SceneTreeItem::SceneTreeItem(const QList<QVariant> &data, SceneTreeItem *parent)
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
void SceneTreeItem::appendChild(SceneTreeItem *item)
{
    childItems.append(item);
}
//! [2]

//! [3]
SceneTreeItem *SceneTreeItem::child(int row)
{
    return childItems.value(row);
}
//! [3]

//! [4]
int SceneTreeItem::childCount() const
{
    return childItems.count();
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
SceneTreeItem *SceneTreeItem::parent()
{
    return parentItem;
}
//! [7]

//! [8]
int SceneTreeItem::row() const
{
    if (parentItem)
        return parentItem->childItems.indexOf(const_cast<SceneTreeItem*>(this));

    return 0;
}
//! [8]
