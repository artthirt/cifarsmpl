#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "cifar_reader.h"

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

private:
	Ui::MainWindow *ui;

	cifar_reader m_cifar;
};

#endif // MAINWINDOW_H
