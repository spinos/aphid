#pragma once

#include <QAbstractItemModel>
#include <QModelIndex>
#include <QVariant>

class SceneTreeItem;
class MlScene;
//! [0]
class SceneTreeModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    SceneTreeModel(MlScene * scene, QObject *parent = 0);
    ~SceneTreeModel();

    QVariant data(const QModelIndex &index, int role) const;
    Qt::ItemFlags flags(const QModelIndex &index) const;
    QVariant headerData(int section, Qt::Orientation orientation,
                        int role = Qt::DisplayRole) const;
    QModelIndex index(int row, int column,
                      const QModelIndex &parent = QModelIndex()) const;
    QModelIndex parent(const QModelIndex &index) const;
    int rowCount(const QModelIndex &parent = QModelIndex()) const;
    int columnCount(const QModelIndex &parent = QModelIndex()) const;

private:
    void setupModelData(SceneTreeItem *parent);
    void addBase(QList<SceneTreeItem*> & parents, const std::string & baseName, int level);
    void addOptions(QList<SceneTreeItem*> & parents);
	void addIntAttr(QList<SceneTreeItem*> & parents, const std::string & attrName, int level, int value);
	SceneTreeItem *rootItem;
	MlScene * m_scene;
};

