#ifndef UAV_SEARCH_CELL_HPP
#define UAV_SEARCH_CELL_HPP
#include <cmath>
#include <vector>
#include <nlohmann/json.hpp>
#include <cadmium/modeling/celldevs/grid/cell.hpp>
#include <cadmium/modeling/celldevs/grid/config.hpp>
#include "uav_search_state.hpp"
using namespace cadmium::celldevs;
static const int DR[9][2] = {{0,0},{-1,0},{-1,1},{0,1},{1,1},{1,0},{1,-1},{0,-1},{-1,-1}};
static const int OPP[9] = {0, 5, 6, 7, 8, 1, 2, 3, 4};
class UAVSearchCell : public GridCell<UAVSearchState, double> {
    double alpha, beta; bool sharedInfo, pinned; std::vector<int> cellId;
public:
    UAVSearchCell(const std::vector<int>& id,
                  const std::shared_ptr<const GridCellConfig<UAVSearchState, double>>& config)
        : GridCell<UAVSearchState, double>(id, config),
          alpha(0.05), beta(0.5), sharedInfo(false), pinned(false), cellId(id) {
        if (config->rawCellConfig.contains("alpha"))      config->rawCellConfig.at("alpha").get_to(alpha);
        if (config->rawCellConfig.contains("beta"))       config->rawCellConfig.at("beta").get_to(beta);
        if (config->rawCellConfig.contains("shared_info")) config->rawCellConfig.at("shared_info").get_to(sharedInfo);
        if (config->rawCellConfig.contains("pinned"))     config->rawCellConfig.at("pinned").get_to(pinned);
        if (alpha <= 0.0 || alpha >= 0.125) alpha = 0.05;
        if (beta  <= 0.0 || beta  > 1.0)   beta  = 0.5;
    }
    [[nodiscard]] UAVSearchState localComputation(
        UAVSearchState state,
        const std::unordered_map<std::vector<int>, NeighborData<UAVSearchState, double>>& neighborhood) const override {
        UAVSearchState next = state;
        if (state.uav == 200) {
            next.uav = 200;
        } else if (state.uav == 100) {
            double bestNeighborProb = 0.0; int bestDir = 1;
            for (int d = 1; d <= 8; d++) {
                std::vector<int> nid = {cellId[0]+DR[d][0], cellId[1]+DR[d][1]};
                auto it = neighborhood.find(nid);
                if (it != neighborhood.end()) {
                    double p = it->second.state->prob;
                    if (p > bestNeighborProb) { bestNeighborProb = p; bestDir = d; }
                }
            }
            if      (bestNeighborProb <= 0.0)        next.uav = 100;
            else if (state.prob >= bestNeighborProb) next.uav = 200;
            else                                     next.uav = bestDir;
        } else if (state.uav == 0) {
            for (const auto& [nid, ndata] : neighborhood) {
                int nCode = ndata.state->uav;
                if (nCode < 1 || nCode > 8) continue;
                int dr = nid[0]-cellId[0], dc = nid[1]-cellId[1];
                for (int d = 1; d <= 8; d++) {
                    if (DR[d][0]==dr && DR[d][1]==dc) {
                        if (nCode==OPP[d]) next.uav=100;
                        break;
                    }
                }
                if (next.uav==100) break;
            }
        } else if (state.uav >= 1 && state.uav <= 8) {
            next.uav = 0;
        }
        // Pinned hotspot: UAV logic above runs normally (UAV can arrive and lock),
        // but probability is always restored to 1.0 — permanent beacon.
        if (pinned) { next.prob = 1.0; return next; }
        double neighborSum = 0.0;
        for (const auto& [nid, ndata] : neighborhood) neighborSum += ndata.state->prob;
        double newProb = (1.0-8.0*alpha)*state.prob + alpha*neighborSum;
        if (state.uav == 100) newProb *= beta;
        if (sharedInfo) {
            for (const auto& [nid, ndata] : neighborhood)
                if (ndata.state->uav==100) { newProb *= 0.85; break; }
        }
        next.prob = std::max(0.0, std::min(1.0, newProb));
        return next;
    }
    [[nodiscard]] double outputDelay(const UAVSearchState& state) const override { return 1.0; }
};
#endif
