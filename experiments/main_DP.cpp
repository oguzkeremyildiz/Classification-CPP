//
// Created by Oğuz Kerem Yıldız on 7.12.2025.
//

#include <iostream>

#include "mpi.h"
#include "DataSet/DataSet.h"
#include "Model/DecisionTree/SoftDecisionTree.h"
#include "Parameter/SoftDecisionTreeParameter.h"
#include <chrono>

using namespace std::chrono;
int main(int argc, char** argv) {
    int p, id;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    fflush(stdout);
    DataSet iris, bupa, dermatology, optdigits, pendigits, wave, yeast;
    vector<AttributeType> attributeTypes;
    for (int i = 0; i < 4; i++) {
        attributeTypes.push_back(AttributeType::CONTINUOUS);
    }
    DataDefinition dataDefinition = DataDefinition(attributeTypes);
    iris = DataSet(dataDefinition, ",", "datasets/iris.data");
    attributeTypes.clear();
    for (int i = 0; i < 6; i++) {
        attributeTypes.push_back(AttributeType::CONTINUOUS);
    }
    dataDefinition = DataDefinition(attributeTypes);
    bupa = DataSet(dataDefinition, ",", "datasets/bupa.data");
    attributeTypes.clear();
    for (int i = 0; i < 34; i++) {
        attributeTypes.push_back(AttributeType::CONTINUOUS);
    }
    dataDefinition = DataDefinition(attributeTypes);
    dermatology = DataSet(dataDefinition, ",", "datasets/dermatology.data");
    attributeTypes.clear();
    for (int i = 0; i < 64; i++) {
        attributeTypes.push_back(AttributeType::CONTINUOUS);
    }
    dataDefinition = DataDefinition(attributeTypes);
    optdigits = DataSet(dataDefinition, ",", "datasets/optdigits.data");
    attributeTypes.clear();
    for (int i = 0; i < 16; i++) {
        attributeTypes.push_back(AttributeType::CONTINUOUS);
    }
    dataDefinition = DataDefinition(attributeTypes);
    pendigits = DataSet(dataDefinition, ",", "datasets/pendigits.data");
    attributeTypes.clear();
    for (int i = 0; i < 21; i++) {
        attributeTypes.push_back(AttributeType::CONTINUOUS);
    }
    dataDefinition = DataDefinition(attributeTypes);
    wave = DataSet(dataDefinition, ",", "datasets/wave.data");
    attributeTypes.clear();
    for (int i = 0; i < 8; i++) {
        attributeTypes.push_back(AttributeType::CONTINUOUS);
    }
    dataDefinition = DataDefinition(attributeTypes);
    yeast = DataSet(dataDefinition, ",", "datasets/yeast.data");
    auto start = high_resolution_clock::now();
    SoftDecisionTreeParameter* softTreeParameter;
    softTreeParameter = new SoftDecisionTreeParameter(1, 0.1, 0.99, 0.2, 100, 0, 32, p, id);
    auto softTree = SoftDecisionTree();
    InstanceList instanceList = dermatology.getInstanceList();
    softTree.train(instanceList, softTreeParameter);
    const double dermatologyError = 100 * softTree.test(dermatology.getInstanceList())->getErrorRate();
    cout << "Dermatology" << id << " " << dermatologyError << endl;
    auto end1 = duration_cast<seconds>(high_resolution_clock::now() - start).count();
    cout << end1 << " seconds."<< endl;
    fflush(stdout);
    start = high_resolution_clock::now();
    softTree = SoftDecisionTree();
    instanceList = iris.getInstanceList();
    softTreeParameter = new SoftDecisionTreeParameter(1, 0.1, 0.99, 0.2, 100, 0, 32, p, id);
    softTree.train(instanceList, softTreeParameter);
    const double irisError = 100 * softTree.test(iris.getInstanceList())->getErrorRate();
    cout << "Iris" << id << " " << irisError << endl;
    auto end2 = duration_cast<seconds>(high_resolution_clock::now() - start).count();
    cout << end2 << " seconds."<< endl;
    fflush(stdout);
    start = high_resolution_clock::now();
    softTreeParameter = new SoftDecisionTreeParameter(1, 0.001, 0.99, 0.2, 100, 0, 32, p, id);
    softTree = SoftDecisionTree();
    instanceList = bupa.getInstanceList();
    softTree.train(instanceList, softTreeParameter);
    const double bupaError = 100 * softTree.test(bupa.getInstanceList())->getErrorRate();
    cout << "Bupa" << id << " " << bupaError << endl;
    auto end3 = duration_cast<seconds>(high_resolution_clock::now() - start).count();
    cout << end3 << " seconds."<< endl;
    fflush(stdout);
    start = high_resolution_clock::now();
    softTreeParameter = new SoftDecisionTreeParameter(1, 0.001, 0.99, 0.2, 10, 0, 32, p, id);
    softTree = SoftDecisionTree();
    instanceList = optdigits.getInstanceList();
    softTree.train(instanceList, softTreeParameter);
    const double optdigitsError = 100 * softTree.test(optdigits.getInstanceList())->getErrorRate();
    cout << "Optdigits" << id << " " << optdigitsError << endl;
    auto end4 = duration_cast<seconds>(high_resolution_clock::now() - start).count();
    cout << end4 << " seconds."<< endl;
    fflush(stdout);
    start = high_resolution_clock::now();
    softTreeParameter = new SoftDecisionTreeParameter(1, 0.001, 0.99, 0.2, 10, 0, 32, p, id);
    softTree = SoftDecisionTree();
    instanceList = pendigits.getInstanceList();
    softTree.train(instanceList, softTreeParameter);
    const double pendigitsError = 100 * softTree.test(pendigits.getInstanceList())->getErrorRate();
    cout << "Pendigits" << id << " " << pendigitsError << endl;
    auto end5 = duration_cast<seconds>(high_resolution_clock::now() - start).count();
    cout << end5 << " seconds."<< endl;
    fflush(stdout);
    start = high_resolution_clock::now();
    softTreeParameter = new SoftDecisionTreeParameter(1, 0.001, 0.99, 0.2, 10, 0, 32, p, id);
    softTree = SoftDecisionTree();
    instanceList = wave.getInstanceList();
    softTree.train(instanceList, softTreeParameter);
    const double waveError = 100 * softTree.test(wave.getInstanceList())->getErrorRate();
    cout << "Wave" << id << " " << waveError << endl;
    auto end6 = duration_cast<seconds>(high_resolution_clock::now() - start).count();
    cout << end6 << " seconds."<< endl;
    fflush(stdout);
    start = high_resolution_clock::now();
    softTreeParameter = new SoftDecisionTreeParameter(1, 0.001, 0.99, 0.2, 100, 0, 32, p, id);
    softTree = SoftDecisionTree();
    instanceList = yeast.getInstanceList();
    softTree.train(instanceList, softTreeParameter);
    const double yeastError = 100 * softTree.test(yeast.getInstanceList())->getErrorRate();
    cout << "Yeast" << id << " " << yeastError << endl;
    auto end7 = duration_cast<seconds>(high_resolution_clock::now() - start).count();
    cout << end7 << " seconds."<< endl;
    fflush(stdout);
    cout << "Total time: " << end1 << " + " << end2 << " + " << end3 << " + " << end4 << " + " << end5 << " + " << end6 << " + " << end7 << " = " << end1 + end2 + end3 + end4 + end5 + end6 + end7 << " seconds."<< endl;
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    if (id == 0) {
        cout << "================ SUMMARY (p = " << p << ") ================" << endl;
        cout << "Dermatology error " << dermatologyError << " time " << end1 << " seconds." << endl;
        cout << "Iris error " << irisError << " time " << end2 << " seconds." << endl;
        cout << "Bupa error " << bupaError << " time " << end3 << " seconds." << endl;
        cout << "Optdigits error " << optdigitsError << " time " << end4 << " seconds." << endl;
        cout << "Pendigits error " << pendigitsError << " time " << end5 << " seconds." << endl;
        cout << "Wave error " << waveError << " time " << end6 << " seconds." << endl;
        cout << "Yeast error " << yeastError << " time " << end7 << " seconds." << endl;
        cout << "Total time " << end1 + end2 + end3 + end4 + end5 + end6 + end7 << " seconds." << endl;
        cout << "=========================================================" << endl;
        fflush(stdout);
    }
    MPI_Finalize();
}
