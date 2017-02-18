#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>

const QString model_file("model.bin");

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
//	cnv.push_back(4);

	cnv_w.push_back(5);
	cnv_w.push_back(5);
//	cnv_w.push_back(3);

	mlp.push_back(900);
	mlp.push_back(800);
	mlp.push_back(700);
	mlp.push_back(400);
	mlp.push_back(10);

	m_train.setCifar(&m_cifar);
	m_train.setConvLayers(cnv, cnv_w, ct::Size(32, 32));
	m_train.setMlpLayers(mlp);

	m_train.init();

	ui->tb_main->setCurrentIndex(0);

	ui->pte_logs->appendPlainText("Current directory: " + m_cifar.currentDirectory());
	ui->pte_logs->appendPlainText("Binary data exists: " + QString(m_cifar.isBinDataExists()? "True" : "False"));
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
	if(!m_cifar.isBinDataExists()){
		ui->pte_logs->appendPlainText("<<<Not exists binary data!>>>");
		return;
	}

	if(ui->pb_pass->isChecked()){
		ui->pte_logs->appendPlainText("<<<Not run test when pass work!>>>");
		return;
	}
	double acc = -1, l2 = -1;
	m_train.getEstimateTest(acc, l2, ui->chb_gpu->isChecked());

	ui->lb_out->setText(QString("Test: acc=%1, l2=%2").arg(acc).arg(l2));
	ui->pte_logs->appendPlainText(QString("Test: acc=%1, l2=%2").arg(acc).arg(l2));
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

	std::vector< double > percents;

	m_train.pass(50, ui->chb_gpu->isChecked(), &percents);

	{
		std::stringstream ss;
		ss << std::setprecision(2);
		for(size_t i = 0; i < percents.size(); ++i){
			ss << percents[i] << " ";
		}
		//ui->pte_logs->appendPlainText(QString(">> ") + ss.str().c_str());
	}

	uint it = ui->chb_gpu->isChecked()? m_train.iteration_gpu() : m_train.iteration();

	ui->lb_it->setText(QString("Iteraion %1").arg(it));

	if((it % 20) == 0){
		m_train.getEstimate(500, acc, l2, ui->chb_gpu->isChecked());

		qDebug("iteration %d: acc=%f, l2=%f", it, acc, l2);

		ui->lb_out->setText(QString("iteration %1: acc=%2, l2=%3").arg(it).arg(acc).arg(l2));
		ui->pte_logs->appendPlainText(QString("iteration %1: acc=%2, l2=%3").arg(it).arg(acc).arg(l2));

		update_prediction();
	}
}

void MainWindow::update_prediction()
{
	QVector<int> pr = m_train.predict(ui->wcifar->output_data(), ui->chb_gpu->isChecked());

	ui->wcifar->updatePredictfromIndex(pr);

	ui->wdgW->set_weightR(m_train.cnvW(0, ui->chb_gpu->isChecked()));
	ui->wdgW->set_weightG(m_train.cnvW(1, ui->chb_gpu->isChecked()));
	ui->wdgW->set_weightB(m_train.cnvW(2, ui->chb_gpu->isChecked()));
	ui->wdgW->update();
}

void MainWindow::on_pb_pass_clicked(bool checked)
{
	if(!m_cifar.isBinDataExists()){
		checked = false;
		ui->pb_pass->setChecked(false);
	}
	m_doIter = checked;
}

void MainWindow::on_chb_gpu_clicked(bool checked)
{
	if(checked && m_cifar.isBinDataExists())
		m_train.init_gpu();
}

void MainWindow::on_dsb_alpha_valueChanged(double arg1)
{
	m_train.setAlpha(arg1);
}

void MainWindow::on_actOpenDir_triggered()
{
	QFileDialog dlg;
	dlg.setAcceptMode(QFileDialog::AcceptOpen);
	dlg.setFileMode(QFileDialog::Directory);

	if(dlg.exec()){
		if(m_cifar.openDir(dlg.directory().absolutePath())){
			ui->wcifar->update_source();
		}
	}
}

void MainWindow::on_action_save_model_triggered()
{
	m_train.saveToFile(model_file, ui->chb_gpu->isChecked());
}

void MainWindow::on_actionLoad_model_triggered()
{
	if(!m_train.loadFromFile(model_file, ui->chb_gpu->isChecked())){
		ui->pte_logs->appendPlainText("Model not loaded");
	}
	ui->pte_logs->appendPlainText("Model loaded");
}
