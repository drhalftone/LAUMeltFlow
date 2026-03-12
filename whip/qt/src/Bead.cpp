#include "Bead.h"
#include <cmath>

Bead::Bead(int id, double mass, double stiffness, double damping,
           double drag, QObject *parent)
    : QObject(parent)
    , m_id(id)
    , m_mass(mass)
    , m_stiffness(stiffness)
    , m_damping(damping)
    , m_drag(drag)
{
}

void Bead::onNeighborState(int /*neighborId*/, QVector2D neighborPos,
                           QVector2D neighborVel, double restLength)
{
    // --- This IS the message-passing step ---
    // Compute the spring + damping force from this one neighbor
    // and add it to our accumulator.

    QVector2D delta = neighborPos - m_pos;
    float dist = delta.length();
    if (dist < 1e-7f)
        return;

    QVector2D dir = delta / dist;
    float stretch = dist - static_cast<float>(restLength);

    // Hooke's law: F = k * stretch * direction
    m_forceAccum += static_cast<float>(m_stiffness) * stretch * dir;

    // Viscous damping along the rod axis
    if (m_damping > 0.0) {
        QVector2D relVel = neighborVel - m_vel;
        float vAlong = QVector2D::dotProduct(relVel, dir);
        m_forceAccum += static_cast<float>(m_damping) * vAlong * dir;
    }
}

void Bead::applyGravity(double gravity)
{
    if (!m_fixed) {
        m_forceAccum += QVector2D(0.0f, -static_cast<float>(m_mass * gravity));

        // Global drag: F_drag = -drag * v  (resists all motion)
        if (m_drag > 0.0)
            m_forceAccum -= static_cast<float>(m_drag) * m_vel;
    }
}

void Bead::integrate(double dt)
{
    if (m_fixed) {
        m_forceAccum = QVector2D(0, 0);
        return;
    }

    float fdt = static_cast<float>(dt);

    // Symplectic Euler: velocity first, then position with new velocity
    QVector2D acc = m_forceAccum / static_cast<float>(m_mass);
    m_vel += fdt * acc;
    m_pos += fdt * m_vel;

    emit stateChanged(m_id, m_pos, m_vel);
}

void Bead::resetForces()
{
    m_forceAccum = QVector2D(0, 0);
}
