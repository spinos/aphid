/*
 *  SceneSceneTreeModel.h
 *  spinboxdelegate
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
 
#pragma once

#include <QAbstractItemModel>
#include <QModelIndex>
#include <QVariant>
#include <map>
class SceneTreeItem;
class MlScene;
//! [0]
class SceneTreeModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    SceneTreeModel(const QStringList &headers, MlScene* scene, QObject *parent = 0);
    ~SceneTreeModel();
//! [0] //! [1]

    QVariant data(const QModelIndex &index, int role) const;
    QVariant headerData(int section, Qt::Orientation orientation,
                        int role = Qt::DisplayRole) const;

    QModelIndex index(int row, int column,
                      const QModelIndex &parent = QModelIndex()) const;
    QModelIndex parent(const QModelIndex &index) const;

    int rowCount(const QModelIndex &parent = QModelIndex()) const;
    int columnCount(const QModelIndex &parent = QModelIndex()) const;
//! [1]

//! [2]
    Qt::ItemFlags flags(const QModelIndex &index) const;
    bool setData(const QModelIndex &index, const QVariant &value,
                 int role = Qt::EditRole);
    bool setHeaderData(int section, Qt::Orientation orientation,
                       const QVariant &value, int role = Qt::EditRole);

    bool insertColumns(int position, int columns,
                       const QModelIndex &parent = QModelIndex());
    bool removeColumns(int position, int columns,
                       const QModelIndex &parent = QModelIndex());
    bool insertRows(int position, int rows,
                    const QModelIndex &parent = QModelIndex());
    bool removeRows(int position, int rows,
                    const QModelIndex &parent = QModelIndex());
					
public slots:
	void receiveData(QWidget * editor);

private:
    void setupModelData(SceneTreeItem *parent);
    SceneTreeItem *getItem(const QModelIndex &index) const;
	void addBase(QList<SceneTreeItem*> & parents, const std::string & baseName, int level);
	void addIntAttr(QList<SceneTreeItem*> & parents, const std::string & attrName, int level, int value);
	void addFltAttr(QList<SceneTreeItem*> & parents, const std::string & attrName, int level, float value);
	void addBolAttr(QList<SceneTreeItem*> & parents, const std::string & attrName, int level, bool value);
	void addRGBAttr(QList<SceneTreeItem*> & parents, const std::string & attrName, int level, QColor value);
	void addOptions(QList<SceneTreeItem*> & parents);
	void addLights(QList<SceneTreeItem*> & parents);
	
	std::map<std::string, int> m_baseRows;
    SceneTreeItem *rootItem;
	MlScene * m_scene;
};

