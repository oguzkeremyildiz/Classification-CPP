//
// Created by Oğuz Kerem Yıldız on 29.11.2025.
//

#include "SoftDecisionNode.h"
#include <cmath>
#include <iostream>
#include <climits>
#include <random>
#include <algorithm>
#include <mpi.h>
#include <unordered_map>

#include "../../../../Attribute/ContinuousAttribute.h"
#include "../../../../Parameter/SoftDecisionTreeParameter.h"

static const int TAG_TASK = 1;
static const int TAG_RESULT = 2;
static const int TAG_STOP = 3;

inline bool kroneckerDelta(const int& r, const int& k) {
    return r == k;
}

inline double getRandomValue() {
    return (rand() + 0.0) / RAND_MAX;
}

void SoftDecisionNode::copyTo(SoftDecisionNode* node) {
    node->w = vector(this->w);
    ((SoftDecisionNode*)node->children.at(0))->w0s = vector(((SoftDecisionNode*)this->children.at(0))->w0s);
    ((SoftDecisionNode*)node->children.at(1))->w0s = vector(((SoftDecisionNode*)this->children.at(1))->w0s);
}

SoftDecisionNode::SoftDecisionNode(SoftDecisionNode* parent, const InstanceList& trainData) {
    this->parent = parent;
    w0s = std::vector<double>(trainData.getDistinctClassLabels().size());
    for (double & w0 : w0s) {
        w0 = (getRandomValue() * 0.02) - 0.01;
    }
    outputs = std::vector<double>(w0s.size());
    for (int i = 0; i < w0s.size(); i++) {
        outputs[i] = (getRandomValue() * 0.02) - 0.01;
    }
    w = std::vector<double>(trainData.get(0)->attributeSize() + 1);
    for (double & i : w) {
        i = (getRandomValue() * 0.02) - 0.01;
    }
    weight = 0.0;
}

void SoftDecisionNode::manager(const InstanceList& cvData, InstanceList& trainData, SoftDecisionTreeParameter* parameter, const unordered_map<string, int>& classLabelIndexMap, int n, int workers) {
    const int weightSize = trainData.get(0)->attributeSize() + 1;
    const int classSize = classLabelIndexMap.size();
    const int resultSize = 1 + weightSize + 2 * classSize;
    std::vector<SoftDecisionNode*> queue;
    std::vector<SoftDecisionNode*> busy(workers + 1, nullptr);
    std::vector<std::vector<double>> taskBuffer(workers + 1);
    std::vector<double> resultBuffer((workers + 1) * resultSize, 0.0);
    std::vector<MPI_Request> sendRequests(workers + 1, MPI_REQUEST_NULL);
    std::vector<MPI_Request> recvRequests(workers + 1, MPI_REQUEST_NULL);
    std::vector<SoftDecisionNode*> stack;
    std::vector<SoftDecisionNode*> ancestors;
    queue.push_back(this);
    int pending = 0;
    while (!queue.empty() || pending > 0) {
        for (int w = 1; w <= workers && !queue.empty(); w++) {
            if (busy[w] != nullptr) {
                continue;
            }
            SoftDecisionNode* node = queue.back();
            queue.pop_back();
            busy[w] = node;
            ancestors.clear();
            for (SoftDecisionNode* up = node->parent; up != nullptr; up = up->parent) {
                ancestors.push_back(up);
            }
            taskBuffer[w].clear();
            taskBuffer[w].push_back(0.0);
            int position = 0;
            int wanted = 0;
            stack.clear();
            stack.push_back(this);
            while (!stack.empty()) {
                SoftDecisionNode* current = stack.back();
                stack.pop_back();
                bool onPath = false;
                for (unsigned int a = 0; a < ancestors.size(); a++) {
                    if (ancestors[a] == current) {
                        onPath = true;
                    }
                }
                if (current == node) {
                    wanted = position;
                }
                position++;
                taskBuffer[w].push_back(onPath ? 1.0 : 0.0);
                for (int i = 0; i < weightSize; i++) {
                    taskBuffer[w].push_back(current->w[i]);
                }
                for (int i = 0; i < classSize; i++) {
                    taskBuffer[w].push_back(current->w0s[i]);
                }
                if (onPath) {
                    stack.push_back((SoftDecisionNode*) current->children.at(1));
                    stack.push_back((SoftDecisionNode*) current->children.at(0));
                }
            }
            taskBuffer[w][0] = wanted;
            MPI_Isend(taskBuffer[w].data(), taskBuffer[w].size(), MPI_DOUBLE, w, TAG_TASK, MPI_COMM_WORLD, &sendRequests[w]);
            MPI_Irecv(&resultBuffer[w * resultSize], resultSize, MPI_DOUBLE, w, TAG_RESULT, MPI_COMM_WORLD, &recvRequests[w]);
            pending++;
        }
        int index;
        MPI_Status status;
        MPI_Waitany(workers + 1, recvRequests.data(), &index, &status);
        pending--;
        MPI_Wait(&sendRequests[index], MPI_STATUS_IGNORE);
        SoftDecisionNode* node = busy[index];
        busy[index] = nullptr;
        const int base = index * resultSize;
        if (resultBuffer[base] != 0.0) {
            node->children = std::vector<DecisionNode*>();
            node->children.push_back(new SoftDecisionNode(node, trainData));
            node->children.push_back(new SoftDecisionNode(node, trainData));
            for (int i = 0; i < weightSize; i++) {
                node->w[i] = resultBuffer[base + 1 + i];
            }
            for (int i = 0; i < classSize; i++) {
                ((SoftDecisionNode*) node->children.at(0))->w0s[i] = resultBuffer[base + 1 + weightSize + i];
                ((SoftDecisionNode*) node->children.at(1))->w0s[i] = resultBuffer[base + 1 + weightSize + classSize + i];
            }
            queue.push_back((SoftDecisionNode*) node->children.at(1));
            queue.push_back((SoftDecisionNode*) node->children.at(0));
        }
    }
    double stopBuffer = 0.0;
    for (int w = 1; w <= workers; w++) {
        MPI_Isend(&stopBuffer, 1, MPI_DOUBLE, w, TAG_STOP, MPI_COMM_WORLD, &sendRequests[w]);
    }
    MPI_Waitall(workers + 1, sendRequests.data(), MPI_STATUSES_IGNORE);
}

void SoftDecisionNode::worker(const InstanceList& cvData, InstanceList& trainData, SoftDecisionTreeParameter* parameter, const unordered_map<string, int>& classLabelIndexMap, int n) {
    const int weightSize = trainData.get(0)->attributeSize() + 1;
    const int classSize = classLabelIndexMap.size();
    const int nodeSize = 1 + weightSize + classSize;
    const int resultSize = 1 + weightSize + 2 * classSize;
    std::vector<double> result(resultSize, 0.0);
    std::vector<SoftDecisionNode*> stack;
    MPI_Request sendRequest = MPI_REQUEST_NULL;
    while (true) {
        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == TAG_STOP) {
            double stop;
            MPI_Recv(&stop, 1, MPI_DOUBLE, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        }
        int length;
        MPI_Get_count(&status, MPI_DOUBLE, &length);
        std::vector<double> task(length, 0.0);
        MPI_Recv(task.data(), length, MPI_DOUBLE, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        const int wanted = (int) task[0];
        auto* localRoot = new SoftDecisionNode(nullptr, trainData);
        SoftDecisionNode* target = nullptr;
        int cursor = 1;
        int position = 0;
        stack.clear();
        stack.push_back(localRoot);
        while (!stack.empty()) {
            SoftDecisionNode* current = stack.back();
            stack.pop_back();
            const bool internal = task[cursor] != 0.0;
            for (int i = 0; i < weightSize; i++) {
                current->w[i] = task[cursor + 1 + i];
            }
            for (int i = 0; i < classSize; i++) {
                current->w0s[i] = task[cursor + 1 + weightSize + i];
            }
            cursor += nodeSize;
            if (position == wanted) {
                target = current;
            }
            position++;
            current->children = std::vector<DecisionNode*>();
            if (internal) {
                current->children.push_back(new SoftDecisionNode(current, trainData));
                current->children.push_back(new SoftDecisionNode(current, trainData));
                stack.push_back((SoftDecisionNode*) current->children.at(1));
                stack.push_back((SoftDecisionNode*) current->children.at(0));
            }
        }
        srand(parameter->getSeed() + wanted);
        for (int i = 0; i < weightSize; i++) {
            target->w[i] = ((rand() + 0.0) / RAND_MAX) * 0.02 - 0.01;
        }
        const bool accepted = target->findBestSoftDecisionTreeSplit(cvData, trainData, parameter, classLabelIndexMap, n);
        MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
        result[0] = accepted ? 1.0 : 0.0;
        if (accepted) {
            for (int i = 0; i < weightSize; i++) {
                result[1 + i] = target->w[i];
            }
            for (int i = 0; i < classSize; i++) {
                result[1 + weightSize + i] = ((SoftDecisionNode*) target->children.at(0))->w0s[i];
                result[1 + weightSize + classSize + i] = ((SoftDecisionNode*) target->children.at(1))->w0s[i];
            }
        }
        MPI_Isend(result.data(), resultSize, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD, &sendRequest);
    }
    MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
}

void SoftDecisionNode::createSoftDecisionTree(const InstanceList& cvData, InstanceList& trainData, SoftDecisionTreeParameter* parameter, const unordered_map<string, int>& classLabelIndexMap, int n, int p, int id) {
    if (p == 1) {
        const int weightSize = trainData.get(0)->attributeSize() + 1;
        const int classSize = classLabelIndexMap.size();
        const int nodeSize = 1 + weightSize + classSize;
        std::vector<SoftDecisionNode*> queue;
        std::vector<SoftDecisionNode*> stack;
        std::vector<SoftDecisionNode*> ancestors;
        std::vector<double> task;
        queue.push_back(this);
        while (!queue.empty()) {
            SoftDecisionNode* node = queue.back();
            queue.pop_back();
            ancestors.clear();
            for (SoftDecisionNode* up = node->parent; up != nullptr; up = up->parent) {
                ancestors.push_back(up);
            }
            task.clear();
            task.push_back(0.0);
            int position = 0;
            int wanted = 0;
            stack.clear();
            stack.push_back(this);
            while (!stack.empty()) {
                SoftDecisionNode* current = stack.back();
                stack.pop_back();
                bool onPath = false;
                for (unsigned int a = 0; a < ancestors.size(); a++) {
                    if (ancestors[a] == current) {
                        onPath = true;
                    }
                }
                if (current == node) {
                    wanted = position;
                }
                position++;
                task.push_back(onPath ? 1.0 : 0.0);
                for (int i = 0; i < weightSize; i++) {
                    task.push_back(current->w[i]);
                }
                for (int i = 0; i < classSize; i++) {
                    task.push_back(current->w0s[i]);
                }
                if (onPath) {
                    stack.push_back((SoftDecisionNode*) current->children.at(1));
                    stack.push_back((SoftDecisionNode*) current->children.at(0));
                }
            }
            task[0] = wanted;
            auto* localRoot = new SoftDecisionNode(nullptr, trainData);
            SoftDecisionNode* target = nullptr;
            int cursor = 1;
            position = 0;
            stack.clear();
            stack.push_back(localRoot);
            while (!stack.empty()) {
                SoftDecisionNode* current = stack.back();
                stack.pop_back();
                const bool internal = task[cursor] != 0.0;
                for (int i = 0; i < weightSize; i++) {
                    current->w[i] = task[cursor + 1 + i];
                }
                for (int i = 0; i < classSize; i++) {
                    current->w0s[i] = task[cursor + 1 + weightSize + i];
                }
                cursor += nodeSize;
                if (position == wanted) {
                    target = current;
                }
                position++;
                current->children = std::vector<DecisionNode*>();
                if (internal) {
                    current->children.push_back(new SoftDecisionNode(current, trainData));
                    current->children.push_back(new SoftDecisionNode(current, trainData));
                    stack.push_back((SoftDecisionNode*) current->children.at(1));
                    stack.push_back((SoftDecisionNode*) current->children.at(0));
                }
            }
            srand(parameter->getSeed() + wanted);
            for (int i = 0; i < weightSize; i++) {
                target->w[i] = ((rand() + 0.0) / RAND_MAX) * 0.02 - 0.01;
            }
            if (target->findBestSoftDecisionTreeSplit(cvData, trainData, parameter, classLabelIndexMap, n)) {
                node->children = std::vector<DecisionNode*>();
                node->children.push_back(new SoftDecisionNode(node, trainData));
                node->children.push_back(new SoftDecisionNode(node, trainData));
                for (int i = 0; i < weightSize; i++) {
                    node->w[i] = target->w[i];
                }
                for (int i = 0; i < classSize; i++) {
                    ((SoftDecisionNode*) node->children.at(0))->w0s[i] = ((SoftDecisionNode*) target->children.at(0))->w0s[i];
                    ((SoftDecisionNode*) node->children.at(1))->w0s[i] = ((SoftDecisionNode*) target->children.at(1))->w0s[i];
                }
                queue.push_back((SoftDecisionNode*) node->children.at(1));
                queue.push_back((SoftDecisionNode*) node->children.at(0));
            }
        }
    } else if (id == 0) {
        manager(cvData, trainData, parameter, classLabelIndexMap, n, p - 1);
    } else {
        worker(cvData, trainData, parameter, classLabelIndexMap, n);
    }
}

std::vector<double> SoftDecisionNode::calculateOutput(SoftDecisionNode* node, const Instance& data) {
    if (node->children.empty()) {
        return node->w0s;
    }
    double v = data.toVector().biased().dotProduct(Vector(node->w));
    v = 1 / (1 + exp(-v));
    ((SoftDecisionNode*)node->children.at(0))->weight = v;
    ((SoftDecisionNode*)node->children.at(1))->weight = 1 - ((SoftDecisionNode*)node->children.at(0))->weight;
    std::vector<double> left_output = calculateOutput((SoftDecisionNode *) node->children.at(0), data);
    std::vector<double> right_output = calculateOutput((SoftDecisionNode *) node->children.at(1), data);
    for (int i = 0; i < node->outputs.size(); i++) {
        node->outputs[i] = ((SoftDecisionNode*)node->children.at(0))->weight * left_output[i] + ((SoftDecisionNode*)node->children.at(1))->weight * right_output[i];
    }
    return node->outputs;
}

int SoftDecisionNode::errorOfSoftDecisionTree(SoftDecisionNode *node, const InstanceList &data, const unordered_map<string, int>& classLabelIndexMap) {
    int count = 0;
    for (int i = 0; i < data.size(); i++) {
        const auto* instance = data.get(i);
        const int classLabel = classLabelIndexMap.at(instance->getClassLabel());
        std::vector<double> o = calculateOutput(node, *instance);
        double max = o[0];
        int index = 0;
        for (int j = 1; j < o.size(); j++) {
            if (max < o[j]) {
                max = o[j];
                index = j;
            }
        }
        if (index != classLabel) {
            count++;
        }
    }
    return count;
}

SoftDecisionNode *SoftDecisionNode::createNode(const InstanceList& data) {
    auto* node = new SoftDecisionNode();
    node->children.push_back(new SoftDecisionNode(node, data));
    node->children.push_back(new SoftDecisionNode(node, data));
    return node;
}

double* SoftDecisionNode::convert(std::vector<double> v) {
    auto* result = new double[v.size()];
    for (int i = 0; i < v.size(); i++) {
        result[i] = v[i];
    }
    return result;
}


bool SoftDecisionNode::findBestSoftDecisionTreeSplit(const InstanceList& cvData, InstanceList& data, SoftDecisionTreeParameter* parameter, const unordered_map<string, int>& classLabelIndexMap, int n) {
    auto tmp = vector(w0s);
    auto root = this;
    while (root->parent != nullptr) {
        root = root->parent;
    }
    const int performanceBefore = errorOfSoftDecisionTree(root, cvData, classLabelIndexMap);
    this->children = std::vector<DecisionNode*>();
    this->children.push_back(new SoftDecisionNode(this, data));
    this->children.push_back(new SoftDecisionNode(this, data));
    auto* bestNode = createNode(data);
    auto* tmpNode = createNode(data);
    this->copyTo(tmpNode);
    int performanceBest = INT_MAX;
    bool result = true;
    for (int i = 0; i < n; i++) {
        const double eta = 10.0 / pow(2, i);
        gradientDescentSoftDecisionTree(root, &data, parameter, eta, classLabelIndexMap);
        const int performanceNow = errorOfSoftDecisionTree(root, cvData, classLabelIndexMap);
        if (performanceNow < performanceBest) {
            performanceBest = performanceNow;
            this->copyTo(bestNode);
        }
        tmpNode->copyTo(this);
    }
    if (performanceBefore <= performanceBest) {
        result = false;
        this->children.clear();
        this->w0s = vector(tmp);
    } else {
        bestNode->copyTo(this);
    }
    return result;
}

void SoftDecisionNode::gradientDescentSoftDecisionTree(SoftDecisionNode *root, InstanceList* data, SoftDecisionTreeParameter *parameter, double eta, const unordered_map<string, int>& classLabelIndexMap) {
    auto previousUpdateLeftw0s = std::vector<double>(classLabelIndexMap.size());
    auto previousUpdateRightw0s = std::vector<double>(classLabelIndexMap.size());
    auto previousUpdatew = std::vector<double>(data->get(0)->attributeSize() + 1);
    auto order = std::vector<int>(data->size());
    for (int i = 0; i < data->size(); i++) {
        order[i] = i;
    }
    double featureValue;
    for (int i = 0; i < parameter->getEpoch(); i++) {
        std::shuffle(order.begin(), order.end(), std::default_random_engine(parameter->getSeed() + i));
        for (int j = 0; j < data->size(); j++) {
            const auto inst = data->get(order[j]);
            std::vector<double> output = calculateOutput(root, *inst);
            double sum = 0.0;
            for (const double t : output) {
                sum += exp(t);
            }
            for (double & t : output) {
                t = exp(t) / sum;
            }
            int r = classLabelIndexMap.at(inst->getClassLabel());
            double multiplier = eta;
            const SoftDecisionNode* tmpNode = this;
            while (tmpNode->parent) {
                multiplier *= tmpNode->weight;
                tmpNode = tmpNode->parent;
            }
            for (int k = 0; k < classLabelIndexMap.size(); k++) {
                ((SoftDecisionNode*)this->children.at(0))->w0s[k] += multiplier * (kroneckerDelta(r, k) - output[k]) * ((SoftDecisionNode*)this->children.at(0))->weight + parameter->getAlpha() * previousUpdateLeftw0s[k];
                previousUpdateLeftw0s[k] = multiplier * (kroneckerDelta(r, k) - output[k]) * ((SoftDecisionNode*)this->children.at(0))->weight;
                ((SoftDecisionNode*)this->children.at(1))->w0s[k] += multiplier * (kroneckerDelta(r, k) - output[k]) * ((SoftDecisionNode*)this->children.at(1))->weight + parameter->getAlpha() * previousUpdateRightw0s[k];
                previousUpdateRightw0s[k] = multiplier * (kroneckerDelta(r, k) - output[k]) * ((SoftDecisionNode*)this->children.at(1))->weight;
            }
            for (int t = 0; t < inst->attributeSize() + 1; t++) {
                if (t != 0) {
                    featureValue = ((ContinuousAttribute*)inst->getAttribute(t - 1))->getValue();
                } else {
                    featureValue = 1;
                }
                sum = 0.0;
                for (int k = 0; k < classLabelIndexMap.size(); k++) {
                    const double delta = multiplier * (kroneckerDelta(r, k) - output[k]) * featureValue * (((SoftDecisionNode*)this->children.at(0))->outputs[k] - ((SoftDecisionNode*)this->children.at(1))->outputs[k]) * ((SoftDecisionNode*)this->children.at(0))->weight * ((SoftDecisionNode*)this->children.at(1))->weight;
                    sum += delta;
                }
                this->w[t] += sum + parameter->getAlpha() * previousUpdatew[t];
                previousUpdatew[t] = sum;
            }
        }
        eta = eta * parameter->getEtaDecrease();
    }
}

SoftDecisionNode::~SoftDecisionNode() {
    for (const DecisionNode* node : children) {
        delete node;
    }
}
