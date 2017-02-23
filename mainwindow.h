#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>

#include "cifar_reader.h"
#include "cifar_train.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit MainWindow(QWidget *parent = 0);
	~MainWindow();

private slots:
	void on_pb_prev_clicked();

	void on_pb_next_clicked();

	void on_pb_test_clicked();

	void onTimeout();

	void on_pb_pass_clicked();

	void on_pb_pass_clicked(bool checked);

	void on_chb_gpu_clicked(bool checked);

	void on_dsb_alpha_valueChanged(double arg1);

	void on_actOpenDir_triggered();

	void on_action_save_model_triggered();

	void on_actionLoad_model_triggered();

	void on_sb_delay_valueChanged(int arg1);

	void on_sb_batch_valueChanged(int arg1);

	void on_sb_iter_numb_valueChanged(int arg1);

	void on_pb_mode_clicked(bool checked);

	void on_pb_update_clicked();

	void on_dsb_alpha_cnv_valueChanged(double arg1);

private:
	Ui::MainWindow *ui;
	QTimer m_timer;
	bool m_doIter;
	uint m_batch;
	uint m_delimiter;
	int m_delay;

	cifar_train m_train;
	cifar_reader m_cifar;

	void pass();
	void update_prediction();
	void update_statistics();
};

#endif // MAINWINDOW_H
