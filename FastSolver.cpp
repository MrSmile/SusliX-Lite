#include <cmath>
#include <algorithm>
#include "Joints.h"
#include <iostream>
#include <cassert>
#include <cstring>
#include <cstddef>
#include <map>

constexpr float eps = 1e-32;

template<int degreeCount, int vectorWidth> struct JointDataFriction
{
    float projMain[2][degreeCount][vectorWidth];
    float projFrict[2][degreeCount][vectorWidth];
    float dstVel[vectorWidth], frictCoeff[vectorWidth];

    float impulseMain[vectorWidth];
    float impulseFrict[vectorWidth];
    float bodyData[2][degreeCount][vectorWidth];
    float *next[2][vectorWidth];

    JointDataFriction()
    {
        std::memset(this, 0, sizeof(*this) - sizeof(next));
        for(int i = 0; i < vectorWidth; i++)next[0][i] = &bodyData[0][0][i];
        for(int i = 0; i < vectorWidth; i++)next[1][i] = &bodyData[1][0][i];
    }

    void Solve() __attribute__((noinline))
    {
        for(int i = 0; i < vectorWidth; i++)
        {
            float deltaMain = impulseMain[i];  impulseMain[i] += dstVel[i];
            for(int j = 0; j < degreeCount; j++)
                impulseMain[i] -= projMain[0][j][i] * bodyData[0][j][i];
            for(int j = 0; j < degreeCount; j++)
                impulseMain[i] -= projMain[1][j][i] * bodyData[1][j][i];
            impulseMain[i] = std::max(0.0f, impulseMain[i]);

            deltaMain = impulseMain[i] - deltaMain;
            for(int j = 0; j < degreeCount; j++)
                bodyData[0][j][i] += projMain[0][j][i] * deltaMain;
            for(int j = 0; j < degreeCount; j++)
                bodyData[1][j][i] += projMain[1][j][i] * deltaMain;

            float deltaFrict = impulseFrict[i];
            for(int j = 0; j < degreeCount; j++)
                impulseFrict[i] -= projFrict[0][j][i] * bodyData[0][j][i];
            for(int j = 0; j < degreeCount; j++)
                impulseFrict[i] -= projFrict[1][j][i] * bodyData[1][j][i];
            float frict = std::min(frictCoeff[i] * impulseMain[i], std::abs(impulseFrict[i]));
            impulseFrict[i] = std::copysign(frict, impulseFrict[i]);

            deltaFrict = impulseFrict[i] - deltaFrict;
            for(int j = 0; j < degreeCount; j++)
                bodyData[0][j][i] += projFrict[0][j][i] * deltaFrict;
            for(int j = 0; j < degreeCount; j++)
                bodyData[1][j][i] += projFrict[1][j][i] * deltaFrict;
        }
        for(int i = 0; i < vectorWidth; i++)
        {
            for(int j = 0; j < degreeCount; j++)
                next[0][i][j * vectorWidth] = bodyData[0][j][i];
            for(int j = 0; j < degreeCount; j++)
                next[1][i][j * vectorWidth] = bodyData[1][j][i];
        }
    }
};

template<int degreeCount, int vectorWidth> struct JointDataSimple
{
    float proj[2][degreeCount][vectorWidth];
    float dstVel[vectorWidth];

    float impulse[vectorWidth];
    float bodyData[2][degreeCount][vectorWidth];
    float *next[2][vectorWidth];

    JointDataSimple()
    {
        std::memset(this, 0, sizeof(*this) - sizeof(next));
        for(int i = 0; i < vectorWidth; i++)next[0][i] = &bodyData[0][0][i];
        for(int i = 0; i < vectorWidth; i++)next[1][i] = &bodyData[1][0][i];
    }

    void Solve() __attribute__((noinline))
    {
        for(int i = 0; i < vectorWidth; i++)
        {
            float delta = impulse[i];  impulse[i] += dstVel[i];
            for(int j = 0; j < degreeCount; j++)
                impulse[i] -= proj[0][j][i] * bodyData[0][j][i];
            for(int j = 0; j < degreeCount; j++)
                impulse[i] -= proj[1][j][i] * bodyData[1][j][i];
            impulse[i] = std::max(0.0f, impulse[i]);

            delta = impulse[i] - delta;
            for(int j = 0; j < degreeCount; j++)
                bodyData[0][j][i] += proj[0][j][i] * delta;
            for(int j = 0; j < degreeCount; j++)
                bodyData[1][j][i] += proj[1][j][i] * delta;
        }
        for(int i = 0; i < vectorWidth; i++)
        {
            for(int j = 0; j < degreeCount; j++)
                next[0][i][j * vectorWidth] = bodyData[0][j][i];
            for(int j = 0; j < degreeCount; j++)
                next[1][i][j * vectorWidth] = bodyData[1][j][i];
        }
    }
};

template<int vectorWidth> void ClearJointData(JointDataFriction<3, vectorWidth> &data, int index)
{
    data.projMain[0][0][index] = data.projMain[0][1][index] = data.projMain[0][2][index] = 0;
    data.projMain[1][0][index] = data.projMain[1][1][index] = data.projMain[1][2][index] = 0;

    data.projFrict[0][0][index] = data.projFrict[0][1][index] = data.projFrict[0][2][index] = 0;
    data.projFrict[1][0][index] = data.projFrict[1][1][index] = data.projFrict[1][2][index] = 0;

    data.dstVel[index] = data.frictCoeff[index] = 0;
    data.impulseMain[index] = data.impulseFrict[index] = 0;

    data.bodyData[0][0][index] = data.bodyData[0][1][index] = data.bodyData[0][2][index] = 0;
    data.bodyData[1][0][index] = data.bodyData[1][1][index] = data.bodyData[1][2][index] = 0;

    data.next[0][index] = &data.bodyData[0][0][index];
    data.next[1][index] = &data.bodyData[1][0][index];
}

template<int vectorWidth> void FillJointData(const ContactJoint &joint,
    JointDataFriction<3, vectorWidth> &data, int index, float *next0, float *next1)
{
    typedef JointDataFriction<3, vectorWidth> Data;  assert(unsigned(index) < vectorWidth);
    ptrdiff_t offs0 = (reinterpret_cast<char *>(next0) - reinterpret_cast<char *>(&data) + 65536 * sizeof(data)) % sizeof(data);
    ptrdiff_t offs1 = (reinterpret_cast<char *>(next1) - reinterpret_cast<char *>(&data) + 65536 * sizeof(data)) % sizeof(data);
    ptrdiff_t beg = offsetof(Data, bodyData), end = beg + sizeof(data.bodyData) - 2 * vectorWidth * sizeof(float);
    assert(offs0 >= beg && offs0 < end);
    assert(offs1 >= beg && offs1 < end);

    float invSqrtM1 = std::sqrt(joint.body1->invMass + eps), invSqrtM2 = std::sqrt(joint.body2->invMass + eps);
    float invSqrtI1 = std::sqrt(joint.body1->invInertia + eps), invSqrtI2 = std::sqrt(joint.body2->invInertia + eps);

    float normMain = std::sqrt(joint.normalLimiter.compInvMass + eps);
    data.projMain[0][0][index] = joint.normalLimiter.normalProjector1.x * invSqrtM1 * normMain;
    data.projMain[0][1][index] = joint.normalLimiter.normalProjector1.y * invSqrtM1 * normMain;
    data.projMain[0][2][index] = joint.normalLimiter.angularProjector1 * invSqrtI1 * normMain;
    data.projMain[1][0][index] = joint.normalLimiter.normalProjector2.x * invSqrtM2 * normMain;
    data.projMain[1][1][index] = joint.normalLimiter.normalProjector2.y * invSqrtM2 * normMain;
    data.projMain[1][2][index] = joint.normalLimiter.angularProjector2 * invSqrtI2 * normMain;

    float normFrict = std::sqrt(joint.frictionLimiter.compInvMass + eps);
    data.projFrict[0][0][index] = joint.frictionLimiter.normalProjector1.x * invSqrtM1 * normFrict;
    data.projFrict[0][1][index] = joint.frictionLimiter.normalProjector1.y * invSqrtM1 * normFrict;
    data.projFrict[0][2][index] = joint.frictionLimiter.angularProjector1 * invSqrtI1 * normFrict;
    data.projFrict[1][0][index] = joint.frictionLimiter.normalProjector2.x * invSqrtM2 * normFrict;
    data.projFrict[1][1][index] = joint.frictionLimiter.normalProjector2.y * invSqrtM2 * normFrict;
    data.projFrict[1][2][index] = joint.frictionLimiter.angularProjector2 * invSqrtI2 * normFrict;

    data.dstVel[index] = joint.normalLimiter.dstVelocity * normMain;
    data.frictCoeff[index] = 0.3f * normMain / normFrict;

    data.impulseMain[index] = joint.normalLimiter.accumulatedImpulse / normMain;
    data.impulseFrict[index] = joint.frictionLimiter.accumulatedImpulse / normFrict;

    data.bodyData[0][0][index] = joint.body1->velocity.x / invSqrtM1;
    data.bodyData[0][1][index] = joint.body1->velocity.y / invSqrtM1;
    data.bodyData[0][2][index] = joint.body1->angularVelocity / invSqrtI1;
    data.bodyData[1][0][index] = joint.body2->velocity.x / invSqrtM2;
    data.bodyData[1][1][index] = joint.body2->velocity.y / invSqrtM2;
    data.bodyData[1][2][index] = joint.body2->angularVelocity / invSqrtI2;

    data.next[0][index] = next0;
    data.next[1][index] = next1;
}

template<int vectorWidth> void ApplyJointData(ContactJoint &joint, const JointDataFriction<3, vectorWidth> &data, int index)
{
    float normMain = std::sqrt(joint.normalLimiter.compInvMass);
    float normFrict = std::sqrt(joint.frictionLimiter.compInvMass);
    joint.normalLimiter.accumulatedImpulse = data.impulseMain[index] * normMain;
    joint.frictionLimiter.accumulatedImpulse = data.impulseFrict[index] * normFrict;
}

template<int vectorWidth> void ApplyBodyData(RigidBody *body, const float *data, const JointDataFriction<3, vectorWidth> *unused)
{
    float invSqrtM = std::sqrt(body->invMass), invSqrtI = std::sqrt(body->invInertia);
    body->velocity.x = data[0 * vectorWidth] * invSqrtM;
    body->velocity.y = data[1 * vectorWidth] * invSqrtM;
    body->angularVelocity = data[2 * vectorWidth] * invSqrtI;
}

template<int vectorWidth> void ClearJointData(JointDataSimple<3, vectorWidth> &data, int index)
{
    data.proj[0][0][index] = data.proj[0][1][index] = data.proj[0][2][index] = 0;
    data.proj[1][0][index] = data.proj[1][1][index] = data.proj[1][2][index] = 0;

    data.dstVel[index] = data.impulse[index] = 0;

    data.bodyData[0][0][index] = data.bodyData[0][1][index] = data.bodyData[0][2][index] = 0;
    data.bodyData[1][0][index] = data.bodyData[1][1][index] = data.bodyData[1][2][index] = 0;

    data.next[0][index] = &data.bodyData[0][0][index];
    data.next[1][index] = &data.bodyData[1][0][index];
}

template<int vectorWidth> void FillJointData(const ContactJoint &joint,
    JointDataSimple<3, vectorWidth> &data, int index, float *next0, float *next1)
{
    typedef JointDataSimple<3, vectorWidth> Data;  assert(unsigned(index) < vectorWidth);
    ptrdiff_t offs0 = (reinterpret_cast<char *>(next0) - reinterpret_cast<char *>(&data) + 65536 * sizeof(data)) % sizeof(data);
    ptrdiff_t offs1 = (reinterpret_cast<char *>(next1) - reinterpret_cast<char *>(&data) + 65536 * sizeof(data)) % sizeof(data);
    ptrdiff_t beg = offsetof(Data, bodyData), end = beg + sizeof(data.bodyData) - 2 * vectorWidth * sizeof(float);
    assert(offs0 >= beg && offs0 < end);
    assert(offs1 >= beg && offs1 < end);

    float invSqrtM1 = std::sqrt(joint.body1->invMass + eps), invSqrtM2 = std::sqrt(joint.body2->invMass + eps);
    float invSqrtI1 = std::sqrt(joint.body1->invInertia + eps), invSqrtI2 = std::sqrt(joint.body2->invInertia + eps);

    float norm = std::sqrt(joint.normalLimiter.compInvMass + eps);
    data.proj[0][0][index] = joint.normalLimiter.normalProjector1.x * invSqrtM1 * norm;
    data.proj[0][1][index] = joint.normalLimiter.normalProjector1.y * invSqrtM1 * norm;
    data.proj[0][2][index] = joint.normalLimiter.angularProjector1 * invSqrtI1 * norm;
    data.proj[1][0][index] = joint.normalLimiter.normalProjector2.x * invSqrtM2 * norm;
    data.proj[1][1][index] = joint.normalLimiter.normalProjector2.y * invSqrtM2 * norm;
    data.proj[1][2][index] = joint.normalLimiter.angularProjector2 * invSqrtI2 * norm;

    data.dstVel[index] = joint.normalLimiter.dstDisplacingVelocity * norm;
    data.impulse[index] = joint.normalLimiter.accumulatedDisplacingImpulse / norm;

    data.bodyData[0][0][index] = joint.body1->displacingVelocity.x / invSqrtM1;
    data.bodyData[0][1][index] = joint.body1->displacingVelocity.y / invSqrtM1;
    data.bodyData[0][2][index] = joint.body1->displacingAngularVelocity / invSqrtI1;
    data.bodyData[1][0][index] = joint.body2->displacingVelocity.x / invSqrtM2;
    data.bodyData[1][1][index] = joint.body2->displacingVelocity.y / invSqrtM2;
    data.bodyData[1][2][index] = joint.body2->displacingAngularVelocity / invSqrtI2;

    data.next[0][index] = next0;
    data.next[1][index] = next1;
}

template<int vectorWidth> void ApplyJointData(ContactJoint &joint, const JointDataSimple<3, vectorWidth> &data, int index)
{
    float invSqrtM1 = std::sqrt(joint.body1->invMass), invSqrtM2 = std::sqrt(joint.body2->invMass);
    float invSqrtI1 = std::sqrt(joint.body1->invInertia), invSqrtI2 = std::sqrt(joint.body2->invInertia);

    joint.body1->displacingVelocity.x = data.bodyData[0][0][index] * invSqrtM1;
    joint.body1->displacingVelocity.y = data.bodyData[0][1][index] * invSqrtM1;
    joint.body1->displacingAngularVelocity = data.bodyData[0][2][index] * invSqrtI1;
    joint.body2->displacingVelocity.x = data.bodyData[1][0][index] * invSqrtM2;
    joint.body2->displacingVelocity.y = data.bodyData[1][1][index] * invSqrtM2;
    joint.body2->displacingAngularVelocity = data.bodyData[1][2][index] * invSqrtI2;

    float norm = std::sqrt(joint.normalLimiter.compInvMass);
    joint.normalLimiter.accumulatedDisplacingImpulse = data.impulse[index] * norm;
}

template<int vectorWidth> void ApplyBodyData(RigidBody *body, const float *data, const JointDataSimple<3, vectorWidth> *unused)
{
    float invSqrtM = std::sqrt(body->invMass), invSqrtI = std::sqrt(body->invInertia);
    body->displacingVelocity.x = data[0 * vectorWidth] * invSqrtM;
    body->displacingVelocity.y = data[1 * vectorWidth] * invSqrtM;
    body->displacingAngularVelocity = data[2 * vectorWidth] * invSqrtI;
}


float CheckSolveImpulse(ContactJoint &joint)
{
    constexpr int index = 0;
    JointDataFriction<3, 1> data;
    FillJointData(joint, data, index, &data.bodyData[0][0][index], &data.bodyData[1][0][index]);
    joint.SolveImpulse();
    data.Solve();

    float err = 0;
    float invSqrtM1 = std::sqrt(joint.body1->invMass), invSqrtM2 = std::sqrt(joint.body2->invMass);
    float invSqrtI1 = std::sqrt(joint.body1->invInertia), invSqrtI2 = std::sqrt(joint.body2->invInertia);
    err += std::abs(joint.body1->velocity.x - data.bodyData[0][0][index] * invSqrtM1);
    err += std::abs(joint.body1->velocity.y - data.bodyData[0][1][index] * invSqrtM1);
    err += std::abs(joint.body1->angularVelocity - data.bodyData[0][2][index] * invSqrtI1);
    err += std::abs(joint.body2->velocity.x - data.bodyData[1][0][index] * invSqrtM2);
    err += std::abs(joint.body2->velocity.y - data.bodyData[1][1][index] * invSqrtM2);
    err += std::abs(joint.body2->angularVelocity - data.bodyData[1][2][index] * invSqrtI2);

    float normMain = std::sqrt(joint.normalLimiter.compInvMass);
    float normFrict = std::sqrt(joint.frictionLimiter.compInvMass);
    err += std::abs(joint.normalLimiter.accumulatedImpulse - data.impulseMain[index] * normMain);
    err += std::abs(joint.frictionLimiter.accumulatedImpulse - data.impulseFrict[index] * normFrict);

    if(!(err < 1e-3))std::cout << "Error " << err << std::endl;
    return err;
}

float CheckSolveDisplacingImpulse(ContactJoint &joint)
{
    constexpr int index = 0;
    JointDataSimple<3, 1> data;
    FillJointData(joint, data, index, &data.bodyData[0][0][index], &data.bodyData[1][0][index]);
    joint.SolveDisplacement();
    data.Solve();

    float err = 0;
    float invSqrtM1 = std::sqrt(joint.body1->invMass), invSqrtM2 = std::sqrt(joint.body2->invMass);
    float invSqrtI1 = std::sqrt(joint.body1->invInertia), invSqrtI2 = std::sqrt(joint.body2->invInertia);
    err += std::abs(joint.body1->displacingVelocity.x - data.bodyData[0][0][index] * invSqrtM1);
    err += std::abs(joint.body1->displacingVelocity.y - data.bodyData[0][1][index] * invSqrtM1);
    err += std::abs(joint.body1->displacingAngularVelocity - data.bodyData[0][2][index] * invSqrtI1);
    err += std::abs(joint.body2->displacingVelocity.x - data.bodyData[1][0][index] * invSqrtM2);
    err += std::abs(joint.body2->displacingVelocity.y - data.bodyData[1][1][index] * invSqrtM2);
    err += std::abs(joint.body2->displacingAngularVelocity - data.bodyData[1][2][index] * invSqrtI2);

    float norm = std::sqrt(joint.normalLimiter.compInvMass);
    err += std::abs(joint.normalLimiter.accumulatedDisplacingImpulse - data.impulse[index] * norm);

    if(!(err < 1e-3))std::cout << "Error " << err << std::endl;
    return err;
}


struct JointData
{
    int index, next[2];

    JointData() : index(-1), next{-1, -1}
    {
    }
};

struct BodyData
{
    int first, last, *prev;

    BodyData() : first(-1), last(-1), prev(&first)
    {
    }
};

template<int log2width> struct JointSorter
{
    static constexpr int width = 1 << log2width, mask = width - 1;
    static constexpr int mask2 = (2 << log2width) - 1;

    ContactJoint *joints;
    std::vector<JointData> jointData;
    std::map<RigidBody *, BodyData> bodyData;
    std::vector<int> queue;
    int nWaiting, current;

    bool Append(const ContactJoint &joint, int index)
    {
        BodyData &body1 = bodyData[joint.body1];
        BodyData &body2 = bodyData[joint.body2];
        if(!((body1.last ^ current) & ~mask2) || !((body2.last ^ current) & ~mask2))
        {
            if(nWaiting >= queue.size())queue.resize(nWaiting + 1);
            queue[nWaiting++] = index;  return false;
        }
        *body1.prev = body1.last = current | 0;
        body1.prev = &jointData[index].next[0];
        *body2.prev = body2.last = current | 1;
        body2.prev = &jointData[index].next[1];
        jointData[index].index = current >> 1;
        return !((current += 2) & mask2);
    }

    void ProcessQueue()
    {
        int n = nWaiting, i = 0;  nWaiting = 0;
        while(i < n)
            if(Append(joints[queue[i]], queue[i]))
            {
                while(i < n)queue[nWaiting++] = queue[i++];
                n = nWaiting;  i = nWaiting = 0;
            }
            else i++;
    }

    JointSorter(ContactJoint *joints_, int nJoints) : joints(joints_), jointData(nJoints), nWaiting(0), current(0)
    {
        for(int i = 0; i < nJoints; i++)
        {
            if(Append(joints[i], i))ProcessQueue();
        }
        for(;;)
        {
            for(; current & mask2; current += 2);  // TODO: mark for clearing
            if(!nWaiting)break;  ProcessQueue();
        }
        for(auto &body : bodyData)*body.second.prev = body.second.first;
        assert(!(current & mask2));  current >>= 1;
    }

    int Size() const
    {
        return current >> log2width;
    }

    template<typename T> static float *DataPointer(T *data, int ref)
    {
        return &data[ref >> (log2width + 1)].bodyData[ref & 1][0][(ref >> 1) & mask];
    }

    template<typename T> void Fill(T *data) const
    {
        for(int i = 0; i < jointData.size(); i++)
            FillJointData<width>(joints[i], data[jointData[i].index >> log2width], jointData[i].index & mask,
                DataPointer(data, jointData[i].next[0]), DataPointer(data, jointData[i].next[1]));
    }

    template<typename T> void Apply(T *data) const
    {
        for(int i = 0; i < jointData.size(); i++)
            ApplyJointData<width>(joints[i], data[jointData[i].index >> log2width], jointData[i].index & mask);
        for(auto &body : bodyData)
            ApplyBodyData<width>(body.first, DataPointer(data, body.second.first), data);
    }

    template<template<int, int> class T> void Solve(int nIter) const
    {
        std::vector<T<3, width>> data(Size());
        Fill(data.data());
        for(int i = 0; i < nIter; i++)
            for(auto &block : data)block.Solve();
        Apply(data.data());
    }
};

void FastSolveJoints(ContactJoint *joints, int nJoints, int nIter, int nDisp)
{
    constexpr int log2width = 3;
    JointSorter<log2width> sorter(joints, nJoints);
    sorter.Solve<JointDataFriction>(nIter);
    sorter.Solve<JointDataSimple>(nDisp);
}