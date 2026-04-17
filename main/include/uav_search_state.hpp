#ifndef UAV_SEARCH_STATE_HPP
#define UAV_SEARCH_STATE_HPP

#include <iostream>
#include <cmath>
#include <nlohmann/json.hpp>

struct UAVSearchState {
    double prob;
    int    uav;
    int    zone;
    double uncertainty;
    double visit_penalty;
    double shared_penalty;

    UAVSearchState()
        : prob(0.0),
          uav(0),
          zone(0),
          uncertainty(1.0),
          visit_penalty(0.0),
          shared_penalty(0.0) {}
};

inline bool operator!=(const UAVSearchState& x, const UAVSearchState& y) {
    return x.uav != y.uav
        || x.zone != y.zone
        || std::abs(x.prob - y.prob) > 1e-9
        || std::abs(x.uncertainty - y.uncertainty) > 1e-9
        || std::abs(x.visit_penalty - y.visit_penalty) > 1e-9
        || std::abs(x.shared_penalty - y.shared_penalty) > 1e-9;
}

inline std::ostream& operator<<(std::ostream& os, const UAVSearchState& s) {
    // Keep this as prob,uav for now so you do not break the Cadmium viewer/log pipeline.
    os << s.prob << "," << s.uav;
    return os;
}

[[maybe_unused]] inline void from_json(const nlohmann::json& j, UAVSearchState& s) {
    if (j.contains("prob"))           j.at("prob").get_to(s.prob);
    if (j.contains("uav"))            j.at("uav").get_to(s.uav);
    if (j.contains("zone"))           j.at("zone").get_to(s.zone);
    if (j.contains("uncertainty"))    j.at("uncertainty").get_to(s.uncertainty);
    if (j.contains("visit_penalty"))  j.at("visit_penalty").get_to(s.visit_penalty);
    if (j.contains("shared_penalty")) j.at("shared_penalty").get_to(s.shared_penalty);
}

#endif