#ifndef UAV_SEARCH_CELL_HPP
#define UAV_SEARCH_CELL_HPP

#include <cmath>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <cadmium/modeling/celldevs/grid/cell.hpp>
#include <cadmium/modeling/celldevs/grid/config.hpp>
#include "uav_search_state.hpp"

using namespace cadmium::celldevs;

// Direction encoding:
// 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW
static const int DR[9][2] = {
    {0,0}, {-1,0}, {-1,1}, {0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1}
};

static const int OPP[9] = {
    0, 5, 6, 7, 8, 1, 2, 3, 4
};

class UAVSearchCell : public GridCell<UAVSearchState, double> {
    double alpha;
    double beta;
    bool sharedInfo;
    bool pinned;
    std::vector<int> cellId;

    // New parameters
    double w_prob;
    double w_unc;
    double w_visit;
    double w_shared;

    double beta_unc;
    double beta_visit;
    double visit_decay;
    double shared_decay;

    double alpha_high;
    double alpha_low;

public:
    UAVSearchCell(
        const std::vector<int>& id,
        const std::shared_ptr<const GridCellConfig<UAVSearchState, double>>& config
    )
        : GridCell<UAVSearchState, double>(id, config),
          alpha(0.05),
          beta(0.5),
          sharedInfo(false),
          pinned(false),
          cellId(id),
          w_prob(0.6),
          w_unc(0.4),
          w_visit(0.25),
          w_shared(0.35),
          beta_unc(0.40),
          beta_visit(0.30),
          visit_decay(0.98),
          shared_decay(0.95),
          alpha_high(0.08),
          alpha_low(0.02) {

        if (config->rawCellConfig.contains("alpha"))        config->rawCellConfig.at("alpha").get_to(alpha);
        if (config->rawCellConfig.contains("beta"))         config->rawCellConfig.at("beta").get_to(beta);
        if (config->rawCellConfig.contains("shared_info"))  config->rawCellConfig.at("shared_info").get_to(sharedInfo);
        if (config->rawCellConfig.contains("pinned"))       config->rawCellConfig.at("pinned").get_to(pinned);

        if (config->rawCellConfig.contains("w_prob"))       config->rawCellConfig.at("w_prob").get_to(w_prob);
        if (config->rawCellConfig.contains("w_unc"))        config->rawCellConfig.at("w_unc").get_to(w_unc);
        if (config->rawCellConfig.contains("w_visit"))      config->rawCellConfig.at("w_visit").get_to(w_visit);
        if (config->rawCellConfig.contains("w_shared"))     config->rawCellConfig.at("w_shared").get_to(w_shared);

        if (config->rawCellConfig.contains("beta_unc"))     config->rawCellConfig.at("beta_unc").get_to(beta_unc);
        if (config->rawCellConfig.contains("beta_visit"))   config->rawCellConfig.at("beta_visit").get_to(beta_visit);
        if (config->rawCellConfig.contains("visit_decay"))  config->rawCellConfig.at("visit_decay").get_to(visit_decay);
        if (config->rawCellConfig.contains("shared_decay")) config->rawCellConfig.at("shared_decay").get_to(shared_decay);

        if (config->rawCellConfig.contains("alpha_high"))   config->rawCellConfig.at("alpha_high").get_to(alpha_high);
        if (config->rawCellConfig.contains("alpha_low"))    config->rawCellConfig.at("alpha_low").get_to(alpha_low);

        if (alpha <= 0.0 || alpha >= 0.125) alpha = 0.05;
        if (beta  <= 0.0 || beta  > 1.0)    beta  = 0.5;

        if (alpha_high <= 0.0 || alpha_high >= 0.125) alpha_high = 0.08;
        if (alpha_low  <= 0.0 || alpha_low  >= 0.125) alpha_low  = 0.02;
    }

    [[nodiscard]] UAVSearchState localComputation(
        UAVSearchState state,
        const std::unordered_map<std::vector<int>, NeighborData<UAVSearchState, double>>& neighborhood
    ) const override {

        UAVSearchState next = state;

        // ------------------------------------------------------------
        // 0. Obstacles: block UAV occupancy and probability entirely
        // ------------------------------------------------------------
        if (state.zone == 3) {
            next.uav = 0;
            next.prob = 0.0;
            next.uncertainty = state.uncertainty * visit_decay;
            next.visit_penalty = state.visit_penalty * visit_decay;
            next.shared_penalty = state.shared_penalty * shared_decay;
            return next;
        }

        // ------------------------------------------------------------
        // 1. UAV logic
        // ------------------------------------------------------------

        // Locked UAV
        if (state.uav == 200) {
            next.uav = 200;
        }

        // Active UAV chooses direction using weighted score
        else if (state.uav == 100) {
            double bestScore = -1e18;
            int bestDir = 0;

            for (int d = 1; d <= 8; d++) {
                std::vector<int> nid = {cellId[0] + DR[d][0], cellId[1] + DR[d][1]};
                auto it = neighborhood.find(nid);
                if (it == neighborhood.end()) continue;

                const auto& ns = *(it->second.state);

                // Skip obstacles
                if (ns.zone == 3) continue;

                // Distance cost: cardinal = 1, diagonal = sqrt(2)
                double distance = (std::abs(DR[d][0]) + std::abs(DR[d][1]) == 2) ? std::sqrt(2.0) : 1.0;

                // Zone bonus / penalty
                double zone_bonus = 1.0;
                if (ns.zone == 1) zone_bonus = 1.4;   // high-value zone
                if (ns.zone == 2) zone_bonus = 0.7;   // low-value zone

                // Main research-level score:
                // probability + uncertainty - revisit/shared penalties, scaled by distance
                double score =
                    zone_bonus *
                    (w_prob * ns.prob + w_unc * ns.uncertainty) / (distance + 1.0)
                    - w_visit * ns.visit_penalty
                    - w_shared * ns.shared_penalty;

                if (score > bestScore) {
                    bestScore = score;
                    bestDir = d;
                }
            }

            // Stay if no valid move
            if (bestDir == 0) {
                next.uav = 100;
            } else {
                // Optional locking behavior if current cell is still best/high enough
                if (state.prob >= 0.95) next.uav = 200;
                else next.uav = bestDir;
            }
        }

        // Empty cell receives UAV if neighboring direction points here
        else if (state.uav == 0) {
            for (const auto& [nid, ndata] : neighborhood) {
                int nCode = ndata.state->uav;
                if (nCode < 1 || nCode > 8) continue;

                int dr = nid[0] - cellId[0];
                int dc = nid[1] - cellId[1];

                for (int d = 1; d <= 8; d++) {
                    if (DR[d][0] == dr && DR[d][1] == dc) {
                        if (nCode == OPP[d]) {
                            next.uav = 100;
                        }
                        break;
                    }
                }
                if (next.uav == 100) break;
            }
        }

        // Direction code clears after movement
        else if (state.uav >= 1 && state.uav <= 8) {
            next.uav = 0;
        }

        // ------------------------------------------------------------
        // 2. Pinned hotspot: keep probability fixed at 1
        // ------------------------------------------------------------
        if (pinned) {
            next.prob = 1.0;

            // Still allow uncertainty/penalty updates
            if (next.uav == 100 || next.uav == 200) {
                next.uncertainty = std::max(0.0, state.uncertainty - beta_unc);
                next.visit_penalty = state.visit_penalty + 1.0;
                if (sharedInfo) next.shared_penalty = state.shared_penalty + 1.0;
            } else {
                next.uncertainty = state.uncertainty * visit_decay;
                next.visit_penalty = state.visit_penalty * visit_decay;
                next.shared_penalty = state.shared_penalty * shared_decay;
            }

            return next;
        }

        // ------------------------------------------------------------
        // 3. Zone-aware diffusion
        // ------------------------------------------------------------
        double a = alpha;
        if (state.zone == 1) a = alpha_high;
        if (state.zone == 2) a = alpha_low;

        double neighborSum = 0.0;
        int neighborCount = 0;

        for (const auto& [nid, ndata] : neighborhood) {
            // Ignore obstacle cells in diffusion
            if (ndata.state->zone == 3) continue;
            neighborSum += ndata.state->prob;
            neighborCount++;
        }

        // For Moore neighborhood we still normalize toward 8-neighbor style behavior.
        // Missing/obstacle neighbors simply contribute 0.
        double newProb = (1.0 - 8.0 * a) * state.prob + a * neighborSum;

        // Visit-based reduction
        if (state.uav == 100 || state.uav == 200) {
            newProb *= (1.0 - beta_visit);
        }

        // Shared-info nearby reduction
        if (sharedInfo) {
            for (const auto& [nid, ndata] : neighborhood) {
                if (ndata.state->uav == 100 || ndata.state->uav == 200) {
                    newProb *= 0.90;
                    break;
                }
            }
        }

        next.prob = std::max(0.0, std::min(1.0, newProb));

        // ------------------------------------------------------------
        // 4. Uncertainty / penalties update
        // ------------------------------------------------------------
        if (next.uav == 100 || next.uav == 200) {
            // Visiting cell reduces uncertainty and increases penalties
            next.uncertainty = std::max(0.0, state.uncertainty - beta_unc);
            next.visit_penalty = state.visit_penalty + 1.0;

            if (sharedInfo) next.shared_penalty = state.shared_penalty + 1.0;
            else next.shared_penalty = state.shared_penalty * shared_decay;
        } else {
            next.uncertainty = state.uncertainty * visit_decay;
            next.visit_penalty = state.visit_penalty * visit_decay;
            next.shared_penalty = state.shared_penalty * shared_decay;
        }

        next.uncertainty = std::max(0.0, std::min(1.0, next.uncertainty));

        return next;
    }

    [[nodiscard]] double outputDelay(const UAVSearchState& state) const override {
        return 1.0;
    }
};

#endif