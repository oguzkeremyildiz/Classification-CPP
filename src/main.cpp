//
// Driver of the soft decision tree experiments.
//
// The parallelization scheme (NF, RP or DP) is chosen when the project is built
// (cmake -DSCHEME=NF|RP|DP). The program is started with mpirun; the number of
// MPI processes is the p of the experiments.
//
//   mpirun -np 4 ./Main                 all seven data sets
//   mpirun -np 4 ./Main iris bupa       only the named data sets
//
// The data sets are read from ./datasets/<name>.data (UCI Machine Learning Repository).
// The error rate is measured on the data set the tree was induced from.
//

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "mpi.h"
#include "DataSet/DataSet.h"
#include "Model/DecisionTree/SoftDecisionTree.h"
#include "Parameter/SoftDecisionTreeParameter.h"

using namespace std;

struct Experiment {
    const char* name;
    int attributeCount;
    double learningRate;
    int epochs;
};

int main(int argc, char** argv) {
    int p, id;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    const vector<Experiment> experiments = {
        {"dermatology", 34, 0.1, 100},
        {"iris", 4, 0.1, 100},
        {"bupa", 6, 0.001, 100},
        {"optdigits", 64, 0.001, 10},
        {"pendigits", 16, 0.001, 10},
        {"wave", 21, 0.001, 10},
        {"yeast", 8, 0.001, 100}
    };
    const vector<string> selected(argv + 1, argv + argc);
    double total = 0.0;
    for (const Experiment& experiment : experiments) {
        if (!selected.empty() && find(selected.begin(), selected.end(), experiment.name) == selected.end()) {
            continue;
        }
        const vector<AttributeType> attributeTypes(experiment.attributeCount, AttributeType::CONTINUOUS);
        const DataDefinition dataDefinition = DataDefinition(attributeTypes);
        DataSet dataSet = DataSet(dataDefinition, ",", string("datasets/") + experiment.name + ".data");
        InstanceList instanceList = dataSet.getInstanceList();
        auto* parameter = new SoftDecisionTreeParameter(1, experiment.learningRate, 0.99, 0.2, experiment.epochs, 0, 32, p, id);
        SoftDecisionTree softTree = SoftDecisionTree();
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        softTree.train(instanceList, parameter);
        MPI_Barrier(MPI_COMM_WORLD);
        const double seconds = MPI_Wtime() - start;
        total += seconds;
        // With NF only rank 0 holds the induced tree; with RP and DP every rank holds the same tree.
        if (id == 0) {
            const double error = 100 * softTree.test(dataSet.getInstanceList())->getErrorRate();
            cout << experiment.name << " (p = " << p << "): error " << error << " %, time " << seconds << " seconds" << endl;
        }
    }
    if (id == 0) {
        cout << "Total time (p = " << p << "): " << total << " seconds" << endl;
    }
    MPI_Finalize();
    return 0;
}
