// Home pose = standing: 180 deg (tick 2048) on every joint except hip_pitch and ankle_pitch
// (33, 35, 43, 45), whose home is 0/360 deg, right on the encoder wrap.
//   ./build/home          status: mode, homing offset, torque, angle vs home (read only)
//   ./build/home setup    33/35/43/45 -> Extended Position mode. Turns their torque OFF: support the robot!
//   ./build/home go       all 12 move to home together (one smooth trajectory, ~30 deg/s), torque on
//   ./build/home jog      direction check, robot hanging in the air: each joint moves +10 deg (raw servo +)
//                         and back; answer y if it matched the sim's + meaning. Lists the reversed joints.
//   ./build/home range    torque OFF on all; move every joint to both mechanical stops; Ctrl+C prints
//                         each joint's min/max vs home (for joint limits)
//   ./build/home calibrate  torque OFF on all; you hold the legs exactly in the home pose (straight), then
//                           each Homing Offset is set so that pose reads home. Corrections over 30 deg are refused.
// Why Extended mode for those four: Position mode only accepts goals 0..4095, so a joint homed at 0 can't
// go below it, and Homing Offset is limited to +-1024 (+-90 deg) there, too little to shift home to 180.
// In Extended mode a goal of -11 (-1 deg) is fine; home_tick() picks the home in the present turn.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <glob.h>
#include <string>
#include <unistd.h>

#include "dynamixel_sdk.h"

constexpr int IDS[] = {31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46};
const char* JOINTS[6] = {"hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"};
constexpr uint16_t MODE = 11, OFFSET = 20, TORQUE = 64, HW_ERROR = 70, PROFILE_VEL = 112, GOAL = 116, PRESENT = 132;
constexpr uint8_t EXTENDED = 4;
constexpr uint32_t SLOW = 30;  // Profile Velocity, 0.229 rpm units: ~0.7 rad/s (jog)
constexpr uint16_t PROFILE_ACC = 108;
constexpr double SPEED_DEG_S = 30;  // home go: the largest move sets the duration

bool wraps(int id) { return id % 10 == 3 || id % 10 == 5; }

// home tick nearest the present position: 2048, or for the wrap joints the multiple of 4096 closest to now
int32_t home_tick(int id, int32_t present) {
    return wraps(id) ? 4096 * int32_t(std::lround(present / 4096.0)) : 2048;
}

volatile std::sig_atomic_t running = 1;

int main(int argc, char** argv) {
    const char* cmd = argc > 1 ? argv[1] : "status";
    if (strcmp(cmd, "status") && strcmp(cmd, "setup") && strcmp(cmd, "go") && strcmp(cmd, "calibrate") && strcmp(cmd, "range") && strcmp(cmd, "jog"))
        return fprintf(stderr, "usage: home [status|setup|go|calibrate|range|jog]\n"), 1;
    glob_t found{};
    glob("/dev/cu.usbserial-*", 0, nullptr, &found);
    const char* dev = found.gl_pathc ? found.gl_pathv[0] : "/dev/cu.usbserial-NOT-FOUND";
    auto* port = dynamixel::PortHandler::getPortHandler(dev);
    auto* bus = dynamixel::PacketHandler::getPacketHandler(2.0);
    if (!port->openPort() || !port->setBaudRate(115200))
        return fprintf(stderr, "cannot open %s (led_monitor still running?)\n", dev), 1;

    if (!strcmp(cmd, "setup")) {
        printf("33/35/43/45 will go limp (hip and ankle pitch). Support the robot, then press Enter.");
        getchar();
        for (int id : IDS) {
            if (!wraps(id)) continue;
            bool ok = bus->write1ByteTxRx(port, id, TORQUE, 0) == COMM_SUCCESS &&
                      bus->write1ByteTxRx(port, id, MODE, EXTENDED) == COMM_SUCCESS;  // EEPROM: needs torque off
            printf("ID %d -> Extended Position mode%s\n", id, ok ? "" : "  FAILED");
        }
    }

    if (!strcmp(cmd, "jog")) {
        // the sim's + per joint (robot_sim/CLAUDE.md sign table): left, right
        const char* PLUS[6][2] = {{"toes turn IN", "toes turn OUT"},   {"leg swings OUT", "leg swings IN"},
                                  {"leg swings BACK", "leg swings FORWARD"}, {"knee BENDS", "knee STRAIGHTENS (bends backward)"},
                                  {"toe goes DOWN", "toe goes UP"},     {"OUTER edge of the foot goes up", "INNER edge of the foot goes up"}};
        constexpr int32_t JOG = 114;  // ticks = 10 deg
        printf("Hang the robot in the air. Each joint moves +10 deg and back.\n");
        std::string reversed;
        for (int id : IDS) {
            bool left = id / 10 == 3;
            printf("\nID %d %s %s: expect  %s.  Enter to move.", id, left ? "left" : "right", JOINTS[id % 10 - 1],
                   PLUS[id % 10 - 1][!left]);
            getchar();
            uint32_t raw = 0;
            bus->read4ByteTxRx(port, id, PRESENT, &raw);
            int32_t home = home_tick(id, int32_t(raw));
            bus->write1ByteTxRx(port, id, TORQUE, 1);  // torque first: it resets the goal
            bus->write4ByteTxRx(port, id, PROFILE_VEL, SLOW);
            bus->write4ByteTxRx(port, id, GOAL, uint32_t(home + JOG));
            usleep(1500000);
            bus->write4ByteTxRx(port, id, GOAL, uint32_t(home));
            printf(" Did it match? [y/n] ");
            char answer[16] = {};
            if (fgets(answer, sizeof answer, stdin) && answer[0] == 'n') reversed += " " + std::to_string(id);
        }
        printf("\nreversed vs the sim:%s\n", reversed.empty() ? " none" : reversed.c_str());
        port->closePort();
        return 0;
    }

    if (!strcmp(cmd, "range")) {
        printf("All 12 servos go limp. Support the robot, then press Enter.");
        getchar();
        for (int id : IDS) bus->write1ByteTxRx(port, id, TORQUE, 0);
        printf("Move each joint by hand to both stops (gently). Ctrl+C when done.\n");
        std::signal(SIGINT, [](int) { running = 0; });
        double lo[12], hi[12];
        std::fill(lo, lo + 12, 1e9), std::fill(hi, hi + 12, -1e9);
        while (running) {
            for (int i = 0; i < 12; i++) {
                uint32_t raw = 0;
                if (bus->read4ByteTxRx(port, IDS[i], PRESENT, &raw) != COMM_SUCCESS) continue;
                double deg = (int32_t(raw) - home_tick(IDS[i], int32_t(raw))) * 360.0 / 4096;
                lo[i] = std::min(lo[i], deg), hi[i] = std::max(hi[i], deg);
            }
            printf("\r ankle 35 %+6.1f..%+6.1f   ankle 45 %+6.1f..%+6.1f", lo[4], hi[4], lo[10], hi[10]), fflush(stdout);
        }
        printf("\n ID  joint        min vs home  max vs home\n");
        for (int i = 0; i < 12; i++) printf(" %d  %-11s  %+8.1f     %+8.1f\n", IDS[i], JOINTS[i % 6], lo[i], hi[i]);
        port->closePort();
        return 0;
    }

    if (!strcmp(cmd, "calibrate")) {
        printf("All 12 servos go limp. Support the robot, then press Enter.");
        getchar();
        for (int id : IDS) bus->write1ByteTxRx(port, id, TORQUE, 0);  // Homing Offset is EEPROM: torque off
        printf("Hold the legs exactly in the home pose (straight, feet flat and parallel), then press Enter.");
        getchar();
        constexpr double MAX_FIX_DEG = 30;
        for (int id : IDS) {
            uint32_t raw = 0, offset = 0;
            if (bus->read4ByteTxRx(port, id, PRESENT, &raw) != COMM_SUCCESS ||
                bus->read4ByteTxRx(port, id, OFFSET, &offset) != COMM_SUCCESS) { printf("ID %d: no reply\n", id); continue; }
            int32_t present = int32_t(raw), fix = home_tick(id, present) - present;  // Present = actual + offset
            if (std::fabs(fix * 360.0 / 4096) > MAX_FIX_DEG) {
                printf("ID %d: %.1f deg off home -> refused (leg not in the home pose?)\n", id, fix * 360.0 / 4096);
                continue;
            }
            bool ok = bus->write4ByteTxRx(port, id, OFFSET, uint32_t(int32_t(offset) + fix)) == COMM_SUCCESS;
            printf("ID %d: correction %+.2f deg, offset %d -> %d%s\n", id, fix * 360.0 / 4096, int32_t(offset),
                   int32_t(offset) + fix, ok ? "" : "  FAILED");
        }
        printf("Torque is off. Stand the robot with  ./build/home go\n");
    }

    int failed = 0;
    printf(" ID  joint        mode       offset  torque     deg  vs home     goal  hw_error\n");
    for (int id : IDS) {
        uint8_t mode = 0, torque = 0, error = 0;
        uint32_t offset = 0, raw = 0, goal = 0;
        // go: a hardware error (e.g. overload) keeps torque off until a reboot; the reboot also re-reads the
        // absolute encoder, so Present Position is read after it
        if (!strcmp(cmd, "go") && bus->read1ByteTxRx(port, id, HW_ERROR, &error) == COMM_SUCCESS && error) {
            printf(" %d  hardware error 0x%02x%s -> reboot\n", id, error, error & 0x20 ? " (OVERLOAD)" : "");
            bus->reboot(port, id);
            usleep(1000000);
        }
        bool ok = bus->read1ByteTxRx(port, id, MODE, &mode) == COMM_SUCCESS &&
                  bus->read4ByteTxRx(port, id, OFFSET, &offset) == COMM_SUCCESS &&
                  bus->read1ByteTxRx(port, id, TORQUE, &torque) == COMM_SUCCESS &&
                  bus->read4ByteTxRx(port, id, PRESENT, &raw) == COMM_SUCCESS &&
                  bus->read4ByteTxRx(port, id, GOAL, &goal) == COMM_SUCCESS &&
                  bus->read1ByteTxRx(port, id, HW_ERROR, &error) == COMM_SUCCESS;
        if (!ok) { printf(" %d  no reply\n", id); failed++; continue; }
        int32_t present = int32_t(raw), home = home_tick(id, present);
        const char* name = mode == 3 ? "position" : mode == EXTENDED ? "extended" : "other";
        printf(" %d  %-11s  %-8s %8d  %-6s %7.1f  %+7.1f  %7.1f  %s\n", id, JOINTS[id % 10 - 1], name, int32_t(offset),
               torque ? "on" : "off", present * 360.0 / 4096, (present - home) * 360.0 / 4096,
               int32_t(goal) * 360.0 / 4096, error & 0x20 ? "OVERLOAD" : error ? "yes" : "");

        if (!strcmp(cmd, "go") && wraps(id) && mode != EXTENDED) {
            printf("     not in Extended mode: run  home setup  first\n");
            failed++;
        }
    }

    // go: torque on everywhere (that resets each Goal to Present: they hold), then stream one smoothstep
    // trajectory to all 12 with Sync Write, so every joint starts and arrives together
    if (!strcmp(cmd, "go") && !failed) {
        int32_t start[12], home[12];
        double longest = 0;
        for (int i = 0; i < 12; i++) {
            uint32_t raw = 0;
            bool ok = bus->write1ByteTxRx(port, IDS[i], TORQUE, 1) == COMM_SUCCESS &&
                      bus->write4ByteTxRx(port, IDS[i], PROFILE_ACC, 0) == COMM_SUCCESS &&
                      bus->write4ByteTxRx(port, IDS[i], PROFILE_VEL, 0) == COMM_SUCCESS &&
                      bus->read4ByteTxRx(port, IDS[i], PRESENT, &raw) == COMM_SUCCESS;  // after torque on
            if (!ok) { printf(" %d  torque on FAILED\n", IDS[i]); failed++; continue; }
            start[i] = int32_t(raw), home[i] = home_tick(IDS[i], start[i]);
            longest = std::max(longest, std::fabs((home[i] - start[i]) * 360.0 / 4096));
        }
        double duration = std::max(1.0, longest / SPEED_DEG_S);
        dynamixel::GroupSyncWrite goals(port, bus, GOAL, 4);
        auto t0 = std::chrono::steady_clock::now();
        for (double t = 0; !failed && t < duration;) {
            t = std::min(duration, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            double x = t / duration, f = x * x * (3 - 2 * x);
            goals.clearParam();
            for (int i = 0; i < 12; i++) {
                uint32_t g = uint32_t(start[i] + int32_t(std::lround((home[i] - start[i]) * f)));
                uint8_t b[4] = {DXL_LOBYTE(DXL_LOWORD(g)), DXL_HIBYTE(DXL_LOWORD(g)), DXL_LOBYTE(DXL_HIWORD(g)),
                                DXL_HIBYTE(DXL_HIWORD(g))};
                goals.addParam(IDS[i], b);
            }
            goals.txPacket();
            usleep(10000);
        }
        if (!failed) printf(" moved all 12 together in %.1f s (largest move %.1f deg)\n", duration, longest);
    }

    // go: wait until every joint is within ARRIVED_DEG of home, so  home go && walk  starts from home
    constexpr double ARRIVED_DEG = 2.0, TIMEOUT_S = 15.0;
    for (double waited = 0, worst = 1e9; !strcmp(cmd, "go") && !failed && worst > ARRIVED_DEG; waited += 0.2) {
        if (waited > TIMEOUT_S) { printf("did not reach home: worst joint %.1f deg off\n", worst); failed++; break; }
        worst = 0;
        for (int id : IDS) {  // 12 reads take ~0.2 s
            uint32_t raw = 0;
            if (bus->read4ByteTxRx(port, id, PRESENT, &raw) != COMM_SUCCESS) continue;
            worst = std::max(worst, std::fabs((int32_t(raw) - home_tick(id, int32_t(raw))) * 360.0 / 4096));
        }
        printf("\r waiting for home: worst joint %.1f deg off", worst), fflush(stdout);
    }
    if (!strcmp(cmd, "go") && !failed) printf("\nat home\n");
    port->closePort();
    return failed != 0;
}
