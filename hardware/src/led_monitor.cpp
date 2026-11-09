// LED blink + current monitor for the 12 leg servos (XM430-W350, Protocol 2.0, 115200 bps, U2D2).
// Torque stays off unless --torque: then each joint holds where it is now, and keeps holding after exit
// (./build/torque off to release).
//   build: cmake -S . -B build && cmake --build build
//   run:   ./build/led_monitor [--torque] [/dev/cu.usbserial-XXXX]      Ctrl+C stops and turns the LEDs off
// Servo xN (N = 1 hip_yaw ... 6 ankle_roll) blinks N times every 5 s, all at the same blink rate.
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <glob.h>
#include <unistd.h>

#include "dynamixel_sdk.h"

constexpr int BAUD = 115200;  // found by dxl_scan (Mac SDK: no custom rates above 115200)
constexpr int N = 12;
constexpr int IDS[N] = {31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46};  // left 3x, right 4x; x1 hip ... x6 ankle
const char* JOINTS[6] = {"hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"};

// XM430 control table
constexpr uint16_t ADDR_LED = 65;
constexpr uint16_t ADDR_STATUS = 126, STATUS_LEN = 21;  // 126 current(2) 128 vel(4) 132 pos(4) .. 144 voltage(2) 146 temp(1)
constexpr double MA_PER_UNIT = 2.69;

constexpr double CYCLE = 5.0, BLINK = 0.5;  // s; LED on for the first half of each BLINK

// standing home pose: hip_pitch and ankle_pitch at 360 deg, the rest at 180 deg
double home_deg(int id) { int j = id % 10; return j == 3 || j == 5 ? 360.0 : 180.0; }

volatile std::sig_atomic_t running = 1;

int main(int argc, char** argv) {
    glob_t found{};  // default: the first U2D2 (FTDI) port
    glob("/dev/cu.usbserial-*", 0, nullptr, &found);
    const char* dev = found.gl_pathc ? found.gl_pathv[0] : "/dev/cu.usbserial-NOT-FOUND";
    bool torque = false;
    for (int a = 1; a < argc; a++) strcmp(argv[a], "--torque") ? (void)(dev = argv[a]) : (void)(torque = true);
    auto* port = dynamixel::PortHandler::getPortHandler(dev);
    auto* bus = dynamixel::PacketHandler::getPacketHandler(2.0);
    if (!port->openPort() || !port->setBaudRate(BAUD)) {
        fprintf(stderr, "cannot open %s at %d bps\n", dev, BAUD);
        return 1;
    }
    std::signal(SIGINT, [](int) { running = 0; });

    for (int id : IDS) {  // hold the present pose: Goal Position := Present Position, then torque on
        if (!torque) break;
        uint32_t pos = 0;
        bool held = bus->read4ByteTxRx(port, id, 132, &pos) == COMM_SUCCESS &&
                    bus->write4ByteTxRx(port, id, 116, pos) == COMM_SUCCESS &&
                    bus->write1ByteTxRx(port, id, 64, 1) == COMM_SUCCESS;
        printf("ID %d torque on: %s\n", id, held ? "holding" : "FAILED");
    }

    dynamixel::GroupSyncWrite leds(port, bus, ADDR_LED, 1);
    auto write_leds = [&](uint8_t* on) {  // one packet for all 12, no replies
        leds.clearParam();
        for (int i = 0; i < N; i++) leds.addParam(IDS[i], &on[i]);
        leds.txPacket();
    };

    uint8_t led[N] = {}, last[N];
    memset(last, 2, N);  // force the first write
    double current[N] = {}, deg[N] = {}, volt[N] = {};
    int temp[N] = {}, alarm[N] = {};
    bool ok[N] = {};
    auto t0 = std::chrono::steady_clock::now();

    // ponytail: reads one servo (~3 ms at 115200) per loop so the LEDs keep their timing; a full
    // sweep takes ~40 ms. Use GroupSyncRead + a higher baud rate for a control loop.
    for (int k = 0; running; k = (k + 1) % N) {
        double t = std::fmod(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), CYCLE);
        for (int i = 0; i < N; i++) led[i] = t < (IDS[i] % 10) * BLINK && std::fmod(t, BLINK) < BLINK / 2;
        if (memcmp(led, last, N)) {
            write_leds(led);
            memcpy(last, led, N);
        }

        uint8_t d[STATUS_LEN], err = 0;
        ok[k] = bus->readTxRx(port, IDS[k], ADDR_STATUS, STATUS_LEN, d, &err) == COMM_SUCCESS;
        if (ok[k]) {
            current[k] = int16_t(d[0] | d[1] << 8) * MA_PER_UNIT;
            deg[k] = int32_t(d[6] | d[7] << 8 | d[8] << 16 | uint32_t(d[9]) << 24) * 360.0 / 4096;
            volt[k] = (d[18] | d[19] << 8) * 0.1;
            temp[k] = d[20];
            alarm[k] = err;
        }
        if (k != N - 1) continue;

        double total = 0;
        if (isatty(STDOUT_FILENO)) printf("\033[H\033[J");
        printf(" ID  joint          mA     deg  vs home    V    C  alarm\n");
        for (int i = 0; i < N; i++) {
            if (!ok[i]) { printf(" %d  %-11s  -- no reply --\n", IDS[i], JOINTS[IDS[i] % 10 - 1]); continue; }
            total += std::fabs(current[i]);
            printf(" %d  %-11s %6.0f  %6.1f  %+7.1f  %4.1f  %3d  %s\n", IDS[i], JOINTS[IDS[i] % 10 - 1], current[i],
                   deg[i], std::remainder(deg[i] - home_deg(IDS[i]), 360.0), volt[i], temp[i], alarm[i] ? "YES" : "");
        }
        printf("TOTAL %.0f mA  (sum of |Present Current|: motor current only; ~0 with torque off)\n", total);
        fflush(stdout);
    }

    memset(led, 0, N);
    write_leds(led);
    port->closePort();
}
