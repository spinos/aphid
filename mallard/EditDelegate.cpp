#include <QtGui>

#include "EditDelegate.h"
#include "SceneTreeItem.h"
#include <AllEdit.h>

EditDelegate::EditDelegate(QObject *parent)
    : QStyledItemDelegate(parent)
{
}

void EditDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option,
                         const QModelIndex &index) const
{
	SceneTreeItem *item = static_cast<SceneTreeItem*>(index.internalPointer());
    if(item->valueType() == 3 && index.column() == 1) {
		QVariant value = index.model()->data(index, Qt::EditRole);
		QColor col = value.value<QColor>();
		painter->fillRect(option.rect, col);
        
    } else {
        QStyledItemDelegate::paint(painter, option, index);
    }
}

QWidget *EditDelegate::createEditor(QWidget *parent,
    const QStyleOptionViewItem & option,
    const QModelIndex & index) const
{
	SceneTreeItem *item = static_cast<SceneTreeItem*>(index.internalPointer());
	QVariant value = index.model()->data(index, Qt::EditRole);
    //qDebug()<<" crt"<<item->parent()->name().c_str();
    QWidget *editor = 0;
	
	switch (item->valueType()) {
		case 0:
			editor = createIntEditor(index, parent);
			break;
		case 1:
			editor = createDoubleEditor(index, parent);
			break;
		case 2:
			editor = createBoolEditor(index, parent);
			break;
		default:
			editor = createColorEditor(index, parent, value.value<QColor>());
			break;
	}
	connect(editor, SIGNAL(editingFinished()),
            this, SLOT(finishEditing())); 
    return editor;
}
//! [1]

//! [2]
void EditDelegate::setEditorData(QWidget *editor,
                                    const QModelIndex &index) const
{
	SceneTreeItem *item = static_cast<SceneTreeItem*>(index.internalPointer());
	QVariant value = index.model()->data(index, Qt::EditRole);
    
	switch (item->valueType()) {
		case 0:
			setIntEditorValue(editor, value); 
			break;
		case 1:
			setDoubleEditorValue(editor, value); 
			break;
		case 2:
			setBoolEditorValue(editor, value); 
			break;
		default:
			setColorEditorValue(editor, value);
			break;
	}
}
//! [2]

//! [3]
void EditDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                   const QModelIndex &index) const
{
    SceneTreeItem *item = static_cast<SceneTreeItem*>(index.internalPointer());
	switch (item->valueType()) {
		case 0:
			model->setData(index, getIntEditorValue(editor), Qt::EditRole);
			break;
		case 1:
			model->setData(index, getDoubleEditorValue(editor), Qt::EditRole);
			break;
		case 2:
			model->setData(index, getBoolEditorValue(editor), Qt::EditRole);
			break;
		default:
			model->setData(index, getColorEditorValue(editor), Qt::EditRole);
			break;
	}
	qDebug()<<"set mdl";
}

QWidget * EditDelegate::createIntEditor(const QModelIndex &index, QWidget *parent) const
{
	return new QIntEdit(index, parent);
}

QWidget * EditDelegate::createDoubleEditor(const QModelIndex &index, QWidget *parent) const
{
	return new QDoubleEdit(index, parent);
}

QWidget * EditDelegate::createBoolEditor(const QModelIndex &index, QWidget *parent) const
{
    return new QBoolEdit(index, parent);
}

QWidget * EditDelegate::createColorEditor(const QModelIndex &index, QWidget *parent, QColor col) const
{
	return new QColorEdit(col, index, parent);
}

void EditDelegate::setIntEditorValue(QWidget *editor, QVariant & value) const
{
	QIntEdit *le = static_cast<QIntEdit*>(editor);
	le->setValue(value.toInt());
}

void EditDelegate::setDoubleEditorValue(QWidget *editor, QVariant & value) const
{
	QDoubleEdit *le = static_cast<QDoubleEdit*>(editor);
	le->setValue(value.toDouble());
}

void EditDelegate::setBoolEditorValue(QWidget *editor, QVariant & value) const
{
	QBoolEdit *le = static_cast<QBoolEdit*>(editor);
	le->setValue(value.toBool());
}

void EditDelegate::setColorEditorValue(QWidget *editor, QVariant & value) const
{
	QColorEdit *le = static_cast<QColorEdit*>(editor);
	le->setColor(value.value<QColor>());
}

QVariant EditDelegate::getIntEditorValue(QWidget *editor) const
{
	QIntEdit *le = static_cast<QIntEdit*>(editor);
	return QVariant(le->value());
}

QVariant EditDelegate::getDoubleEditorValue(QWidget *editor) const
{
	QDoubleEdit *le = static_cast<QDoubleEdit*>(editor);
	return QVariant(le->value());
}

QVariant EditDelegate::getBoolEditorValue(QWidget *editor) const
{
	QBoolEdit *le = static_cast<QBoolEdit*>(editor);
	return QVariant(le->value());
}

QVariant EditDelegate::getColorEditorValue(QWidget *editor) const
{
	QColorEdit *le = static_cast<QColorEdit*>(editor);
	le->pickColor();
	return QVariant(le->color());
}

void EditDelegate::finishEditing()
{
	QModelEdit * editor = static_cast<QModelEdit *>(sender());
	emit commitData(editor);
}