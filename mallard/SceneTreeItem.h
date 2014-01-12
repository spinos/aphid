#pragma once

#include <QList>
#include <QVariant>

//! [0]
class SceneTreeItem
{
public:
    SceneTreeItem(const QList<QVariant> &data, SceneTreeItem *parent = 0);
    ~SceneTreeItem();

    void appendChild(SceneTreeItem *child);

    SceneTreeItem *child(int row);
    int childCount() const;
    int columnCount() const;
    QVariant data(int column) const;
    int row() const;
    SceneTreeItem *parent();

private:
    QList<SceneTreeItem*> childItems;
    QList<QVariant> itemData;
    SceneTreeItem *parentItem;
};

