#ifndef WIDGETMNIST_H
#define WIDGETMNIST_H

#include <QWidget>
#include <QFile>
#include <QByteArray>
#include <QVector>
#include <QTimer>

#include "cifar_reader.h"

namespace Ui {
class WidgetCifar;
}

class WidgetCifar : public QWidget
{
	Q_OBJECT

public:
	enum {TEST, TRAIN};

	explicit WidgetCifar(QWidget *parent = 0);
	~WidgetCifar();
	/**
	 * @brief setTestMode
	 */
	void setTestMode();
	/**
	 * @brief setTrainMode
	 */
	void setTrainMode();
	/**
	 * @brief mode
	 * @return TEST or TRAIN
	 */
	int mode() const;
	/**
	 * @brief setMnist
	 * set ref to reader mnist data
	 * @param mnist
	 */
	void setCifar(cifar_reader* val);
	/**
	 * @brief index
	 * @return index of beginning of the representation
	 */
	double index() const;
	/**
	 * @brief updatePredictfromIndex
	 * update predict values from index
	 * @param index
	 * @param predict - array of predicted values
	 */
	void updatePredictfromIndex(const QVector<int> &predict, int index = 0);
	/**
	 * @brief next
	 */
	void next();
	/**
	 * @brief prev
	 */
	void prev();
	/**
	 * @brief toBegin
	 */
	void toBegin();

	size_t count() const;

	const QVector<TData> &output_data() const;

	void update_source();

public slots:
	void onTimeout();
	void onTimeoutUpdate();

private:
	Ui::WidgetCifar *ui;
	QTimer m_timer;
	QTimer m_timer_update;
	int m_mode;

	QVector< int > m_prediction_test;
	QVector< int > m_prediction_train;

	QVector<TData> m_output_data;

	cifar_reader* m_cifar;

	double m_index;
	QPoint m_point;
	bool m_update;
	int m_cols;
	int m_start_show;

	QImage m_sel_image;

	// QWidget interface
protected:
	void paintEvent(QPaintEvent *event);
	void resizeEvent(QResizeEvent *event);


	// QWidget interface
protected:
	void mouseMoveEvent(QMouseEvent *event);
};

#endif // WIDGETMNIST_H
