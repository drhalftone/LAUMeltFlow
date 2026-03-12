#pragma once

#include <QMainWindow>
#include <QTimer>

class ChainSimulator;
class ChainWidget;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

private slots:
    void tick();

private:
    ChainSimulator *m_sim;
    ChainWidget *m_view;
    QTimer *m_timer;

    double m_dt = 0.0001;          // simulation timestep (s)
    double m_gravity = 9.81;
    int m_subStepsPerFrame = 100;   // sim steps per render frame
    double m_simTime = 0.0;
};
