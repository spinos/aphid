#include <QtGui>

#include "EditDelegate.h"
#include "SceneTreeItem.h"
#include <QColorEdit.h>

//! [0]
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
    
    QWidget *editor = 0;
	
	switch (item->valueType()) {
		case 0:
			editor = createIntEditor(parent);
			break;
		case 1:
			editor = createDoubleEditor(parent);
			break;
		case 2:
			editor = createBoolEditor(parent);
			break;
		default:
			editor = createColorEditor(parent, value.value<QColor>());
			break;
	}
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
}

QString EditDelegate::translateBoolToStr(const QVariant & src) const
{
	if(src.toBool()) return tr("true");
	return tr("false");
}

QVariant EditDelegate::translateStrToBool(const QString & src) const
{
	if(src == "on" || src == "1" || src == "true") return QVariant(true);
	return QVariant(false);
}

QWidget * EditDelegate::createIntEditor(QWidget *parent) const
{
	QLineEdit *editor = new QLineEdit(parent);
	editor->setValidator(&m_intValidate);
    return editor;
}

QWidget * EditDelegate::createDoubleEditor(QWidget *parent) const
{
	QLineEdit *editor = new QLineEdit(parent);
	editor->setValidator(&m_dlbValidate);
    return editor;
}

QWidget * EditDelegate::createBoolEditor(QWidget *parent) const
{
    return new QLineEdit(parent);
}

QWidget * EditDelegate::createColorEditor(QWidget *parent, QColor col) const
{
	return new QColorEdit(col, parent);
}

void EditDelegate::setIntEditorValue(QWidget *editor, QVariant & value) const
{
	QString t;
	int intValue = value.toInt();
	t.setNum(intValue);
	QLineEdit *le = static_cast<QLineEdit*>(editor);
	le->setText(t);
}

void EditDelegate::setDoubleEditorValue(QWidget *editor, QVariant & value) const
{
	QString t;
	double dlbValue = value.toDouble();
	t.setNum(dlbValue);
	QLineEdit *le = static_cast<QLineEdit*>(editor);
	le->setText(t);
}

void EditDelegate::setBoolEditorValue(QWidget *editor, QVariant & value) const
{
	QString t = translateBoolToStr(value);
	QLineEdit *le = static_cast<QLineEdit*>(editor);
	le->setText(t);
}

void EditDelegate::setColorEditorValue(QWidget *editor, QVariant & value) const
{
	QColorEdit *le = static_cast<QColorEdit*>(editor);
	le->setColor(value.value<QColor>());
}

QVariant EditDelegate::getIntEditorValue(QWidget *editor) const
{
	QLineEdit *le = static_cast<QLineEdit*>(editor);
	int value = le->text().toInt();
	return QVariant(value);
}

QVariant EditDelegate::getDoubleEditorValue(QWidget *editor) const
{
	QLineEdit *le = static_cast<QLineEdit*>(editor);
	double value = le->text().toDouble();
	return QVariant(value);
}

QVariant EditDelegate::getBoolEditorValue(QWidget *editor) const
{
	QLineEdit *le = static_cast<QLineEdit*>(editor);
	return QVariant(translateStrToBool(le->text()));
}

QVariant EditDelegate::getColorEditorValue(QWidget *editor) const
{
	QColorEdit *le = static_cast<QColorEdit*>(editor);
	le->pickColor();
	return QVariant(le->color());
}
