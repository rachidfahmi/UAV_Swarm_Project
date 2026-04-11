#include <cadmium/modeling/celldevs/grid/coupled.hpp>
#include <cadmium/simulation/logger/csv.hpp>
#include <cadmium/simulation/root_coordinator.hpp>
#include <string>
#include "uav_search_cell.hpp"

using namespace cadmium::celldevs;

std::shared_ptr<GridCell<UAVSearchState, double>> addCell(
    const std::vector<int>& cellId,
    const std::shared_ptr<const GridCellConfig<UAVSearchState, double>>& cellConfig) {
    auto model = cellConfig->cellModel;
    if (model == "default" || model == "uav_search")
        return std::make_shared<UAVSearchCell>(cellId, cellConfig);
    throw std::bad_typeid();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0]
                  << " SCENARIO_CONFIG.json [SIM_TIME (default: 50)]" << std::endl;
        return -1;
    }
    std::string configFilePath = argv[1];
    double simTime = (argc > 2) ? std::stod(argv[2]) : 50;

    auto model = std::make_shared<GridCellDEVSCoupled<UAVSearchState, double>>(
        "uav_search", addCell, configFilePath);
    model->buildModel();

    auto rootCoordinator = cadmium::RootCoordinator(model);
    rootCoordinator.setLogger<cadmium::CSVLogger>("output/uav_log.csv", ";");
    rootCoordinator.start();
    rootCoordinator.simulate(simTime);
    rootCoordinator.stop();

    std::cout << "Done. Output: output/uav_log.csv" << std::endl;
    return 0;
}
