#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	m_doIter = false;

	connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimeout()));
	m_timer.start(5);

	ui->wcifar->setCifar(&m_cifar);

	std::vector< int > cnv;
	std::vector< int > mlp;
	std::vector< int > cnv_w;

	cnv.push_back(5);
	cnv.push_back(5);
	cnv.push_back(5);

	cnv_w.push_back(5);
	cnv_w.push_back(3);
	cnv_w.push_back(3);

	mlp.push_back(900);
	mlp.push_back(700);
	mlp.push_back(600);
	mlp.push_back(700);
	mlp.push_back(400);
	mlp.push_back(10);

	m_train.setCifar(&m_cifar);
	m_train.setConvLayers(cnv, cnv_w, ct::Size(32, 32));
	m_train.setMlpLayers(mlp);

	m_train.init();

	ui->tb_main->setCurrentIndex(0);
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::on_pb_prev_clicked()
{
	ui->wcifar->prev();
}

void MainWindow::on_pb_next_clicked()
{
	ui->wcifar->next();
}

void MainWindow::on_pb_test_clicked()
{
	ct::Matf y;
	std::vector< ct::Matf > X;

	m_cifar.getTrain(100, X, y);

	qDebug("X[0] size=(%d, %d)", X[0].rows, X[0].cols);
}

void MainWindow::onTimeout()
{
	if(m_doIter){
		pass();
	}
}

void MainWindow::on_pb_pass_clicked()
{

}

void MainWindow::pass()
{
	double l2, acc;
	m_train.pass(50, ui->chb_gpu->isChecked());

	uint it = ui->chb_gpu->isChecked()? m_train.iteration_gpu() : m_train.iteration();

	ui->lb_it->setText(QString("Iteraion %1").arg(it));

	if((it % 20) == 0){
		m_train.getEstimage(500, acc, l2, ui->chb_gpu->isChecked());

		qDebug("iteration %d: acc=%f, l2=%f", it, acc, l2);

		ui->lb_out->setText(QString("iteration %1: acc=%2, l2=%3").arg(it).arg(acc).arg(l2));

		update_prediction();
	}
}

void MainWindow::update_prediction()
{
	QVector<int> pr = m_train.predict(ui->wcifar->index(), ui->wcifar->count(), ui->chb_gpu->isChecked());

	ui->wcifar->updatePredictfromIndex(pr);

	ui->wdgW->set_weightR(m_train.cnvW(0, ui->chb_gpu->isChecked()));
	ui->wdgW->set_weightG(m_train.cnvW(1, ui->chb_gpu->isChecked()));
	ui->wdgW->set_weightB(m_train.cnvW(2, ui->chb_gpu->isChecked()));
	ui->wdgW->update();
}

void MainWindow::on_pb_pass_clicked(bool checked)
{
	m_doIter = checked;
}

void MainWindow::on_chb_gpu_clicked(bool checked)
{
	if(checked)
		m_train.init_gpu();
}
