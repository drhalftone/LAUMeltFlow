#pragma once

#include <QObject>
#include <QVector2D>

/// A single mass node in the bead chain.
///
/// Each Bead is a QObject that communicates with its neighbors via
/// signals and slots — mirroring GNN message passing:
///   - onNeighborState() = edge message function
///   - m_forceAccum      = aggregation (sum)
///   - integrate()       = node update function
///
class Bead : public QObject {
    Q_OBJECT

public:
    explicit Bead(int id, double mass, double stiffness, double damping,
                  double drag = 0.0, QObject *parent = nullptr);

    int id() const { return m_id; }
    QVector2D pos() const { return m_pos; }
    QVector2D vel() const { return m_vel; }
    double mass() const { return m_mass; }
    bool isFixed() const { return m_fixed; }

    void setPos(const QVector2D &p) { m_pos = p; }
    void setVel(const QVector2D &v) { m_vel = v; }
    void setFixed(bool f) { m_fixed = f; }

signals:
    /// Broadcast state to connected neighbors after integration.
    void stateChanged(int id, QVector2D pos, QVector2D vel);

public slots:
    /// Receive a neighbor's state and accumulate the spring/damping force.
    /// This is the "message" arriving along an edge.
    void onNeighborState(int neighborId, QVector2D neighborPos,
                         QVector2D neighborVel, double restLength);

    /// Apply gravity to the force accumulator.
    void applyGravity(double gravity);

    /// Integrate one timestep (symplectic Euler) and emit stateChanged.
    void integrate(double dt);

    /// Reset force accumulator for the next timestep.
    void resetForces();

private:
    int m_id;
    double m_mass;
    double m_stiffness;
    double m_damping;
    double m_drag;       // global velocity drag coefficient
    bool m_fixed = false;

    QVector2D m_pos;
    QVector2D m_vel;
    QVector2D m_forceAccum;
};
