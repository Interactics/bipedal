// Plays the ZMP walk exported from simulation (sim/export_walk.py -> hardware/motions/walk.csv) on the robot.
//   ./build/walk [--speed 0.5] [--until T] [motions/walk.csv]   start at home (./build/home go); stops at plan time T
//   (or Ctrl+C) and holds. Plan: 0-1 s crouch, 1-2 s still, then steps (0.5 s each, first shift ~2.0-2.5 s).
// Each loop: one Sync Write of all 12 Goal Positions (~6.4 ms at 115200 bps), then one servo's Present
// Current, round robin (~16 ms: the Mac FTDI latency timer) -> ~43 Hz goals, each current every ~0.3 s.
// Goals follow the wall clock, so the walk keeps its speed. Sum of |current| > CURRENT_LIMIT -> torque off.
// goal tick = home tick + sign * command * 4096 / 2pi. sim + is CCW looking at the horn = XM Normal drive +,
// except hip_roll and ankle_pitch (32, 35, 42, 45): on the robot they turn opposite to the sim, sign = -1.
// First run it with the robot hanging in the air and compare with the simulation.
#include <chrono>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <glob.h>
#include <sstream>
#include <string>
#include <vector>

#include "dynamixel_sdk.h"

constexpr uint16_t TORQUE = 64, PROFILE_ACC = 108, PROFILE_VEL = 112, GOAL = 116, CURRENT = 126, PRESENT = 132;
constexpr double DT = 0.01, TICKS_PER_RAD = 4096 / (2 * M_PI), MAX_START_ERROR_DEG = 5;
constexpr double CURRENT_LIMIT = 5.0, AMPS_PER_UNIT = 2.69e-3;  // A, all 12 servos together

volatile std::sig_atomic_t running = 1;

// hip_roll (32, 42) and ankle_pitch (35, 45) turn opposite to the sim: measured with  home jog
int sign(int id) { return id % 10 == 2 || id % 10 == 5 ? -1 : 1; }

int32_t home_tick(int id, int32_t present) {  // same rule as home.cpp
    return id % 10 == 3 || id % 10 == 5 ? 4096 * int32_t(std::lround(present / 4096.0)) : 2048;
}

int main(int argc, char** argv) {
    double speed = 1.0, until = 1e9;
    std::string file = "motions/walk.csv";  // run from hardware/
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--speed") && a + 1 < argc) speed = atof(argv[++a]);
        else if (!strcmp(argv[a], "--until") && a + 1 < argc) until = atof(argv[++a]);
        else file = argv[a];
    }

    std::ifstream in(file);
    if (!in) return fprintf(stderr, "no %s: run  python sim/export_walk.py  in robot_sim\n", file.c_str()), 1;
    std::string line, cell;
    std::getline(in, line);
    std::vector<int> ids;
    std::stringstream header(line);
    std::getline(header, cell, ',');  // "t"
    while (std::getline(header, cell, ',')) ids.push_back(std::stoi(cell));
    std::vector<std::vector<double>> rows;
    while (std::getline(in, line)) {
        std::stringstream row(line);
        std::vector<double> q;
        std::getline(row, cell, ',');  // t: rows are DT apart
        while (std::getline(row, cell, ',')) q.push_back(std::stod(cell));
        rows.push_back(q);
    }
    const size_t n = ids.size();

    glob_t found{};
    glob("/dev/cu.usbserial-*", 0, nullptr, &found);
    const char* dev = found.gl_pathc ? found.gl_pathv[0] : "/dev/cu.usbserial-NOT-FOUND";
    auto* port = dynamixel::PortHandler::getPortHandler(dev);
    auto* bus = dynamixel::PacketHandler::getPacketHandler(2.0);
    if (!port->openPort() || !port->setBaudRate(115200))
        return fprintf(stderr, "cannot open %s (another program using it?)\n", dev), 1;

    // safety: every servo answers, has torque on and stands at home
    std::vector<int32_t> home(n);
    for (size_t i = 0; i < n; i++) {
        uint32_t raw = 0;
        uint8_t torque = 0;
        if (bus->read4ByteTxRx(port, ids[i], PRESENT, &raw) != COMM_SUCCESS ||
            bus->read1ByteTxRx(port, ids[i], TORQUE, &torque) != COMM_SUCCESS)
            return fprintf(stderr, "ID %d: no reply\n", ids[i]), 1;
        home[i] = home_tick(ids[i], int32_t(raw));
        double off = (int32_t(raw) - home[i]) * 360.0 / 4096;
        if (!torque || std::fabs(off) > MAX_START_ERROR_DEG)
            return fprintf(stderr, "ID %d: torque %s, %.1f deg from home -> run  ./build/home go  first\n", ids[i],
                           torque ? "on" : "off", off), 1;
        // no motion profile: the goal stream itself is the trajectory (home go leaves a slow profile)
        bus->write4ByteTxRx(port, ids[i], PROFILE_ACC, 0);
        bus->write4ByteTxRx(port, ids[i], PROFILE_VEL, 0);
    }
    std::signal(SIGINT, [](int) { running = 0; });

    dynamixel::GroupSyncWrite goals(port, bus, GOAL, 4);
    printf("playing %s: %.1f s at speed %.2f (Ctrl+C stops and holds)\n", file.c_str(), rows.size() * DT, speed);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<double> amps(n, 0.0), peaks(n, 0.0);
    double t = 0, total = 0, peak = 0, last_print = -1;
    long loops = 0;
    bool tripped = false;
    for (size_t r = 0; running; r = (r + 1) % n, loops++) {
        t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * speed;  // plan time
        double k = t / DT;
        if (k >= rows.size() - 1 || t >= until) break;
        size_t j = size_t(k);
        double f = k - j;
        goals.clearParam();
        for (size_t i = 0; i < n; i++) {
            double q = rows[j][i] * (1 - f) + rows[j + 1][i] * f;
            uint32_t goal = uint32_t(home[i] + int32_t(std::lround(sign(ids[i]) * q * TICKS_PER_RAD)));
            uint8_t bytes[4] = {DXL_LOBYTE(DXL_LOWORD(goal)), DXL_HIBYTE(DXL_LOWORD(goal)),
                                DXL_LOBYTE(DXL_HIWORD(goal)), DXL_HIBYTE(DXL_HIWORD(goal))};
            goals.addParam(ids[i], bytes);
        }
        goals.txPacket();

        uint16_t raw = 0;
        if (bus->read2ByteTxRx(port, ids[r], CURRENT, &raw) == COMM_SUCCESS) amps[r] = std::fabs(int16_t(raw) * AMPS_PER_UNIT);
        peaks[r] = std::max(peaks[r], amps[r]);
        total = 0;
        for (double a : amps) total += a;
        peak = std::max(peak, total);
        if (total > CURRENT_LIMIT) {  // holding would keep drawing current: only torque off limits it
            for (int id : ids) bus->write1ByteTxRx(port, id, TORQUE, 0);
            tripped = true;
            break;
        }
        if (t - last_print >= 1.0) last_print = t, printf("\r t = %5.1f s   total %.2f A", t, total), fflush(stdout);
    }
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (tripped) {
        printf("\nOVER CURRENT: %.2f A > %.1f A at t = %.1f s -> torque OFF on all servos. Per servo [A]:", total,
               CURRENT_LIMIT, t);
        for (size_t i = 0; i < n; i++) printf(" %d:%.2f", ids[i], amps[i]);
        printf("\n");
    } else {
        printf("\n%s at t = %.1f s, holding the last pose (./build/home go returns to home).\n", running ? "done" : "stopped", t);
    }
    printf("peak total current %.2f A | %.0f goal updates/s\npeak current per servo [A]:", peak, loops / wall);
    for (size_t i = 0; i < n; i++) printf(" %d:%.2f", ids[i], peaks[i]);
    printf("\n");
    port->closePort();
}
