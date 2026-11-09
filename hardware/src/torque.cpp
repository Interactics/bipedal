// Torque on/off for the 12 leg servos (XM430, Protocol 2.0, 115200 bps).
//   ./build/torque on    each joint holds where it is now (Goal Position := Present Position first)
//   ./build/torque off   all joints go limp: support the robot first!
#include <cstdio>
#include <cstring>
#include <glob.h>

#include "dynamixel_sdk.h"

constexpr int IDS[] = {31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46};
constexpr uint16_t ADDR_TORQUE = 64, ADDR_GOAL = 116, ADDR_PRESENT = 132;

int main(int argc, char** argv) {
    if (argc < 2 || (strcmp(argv[1], "on") && strcmp(argv[1], "off"))) return fprintf(stderr, "usage: torque on|off [port]\n"), 1;
    bool on = !strcmp(argv[1], "on");
    glob_t found{};
    glob("/dev/cu.usbserial-*", 0, nullptr, &found);
    const char* dev = argc > 2 ? argv[2] : found.gl_pathc ? found.gl_pathv[0] : "/dev/cu.usbserial-NOT-FOUND";
    auto* port = dynamixel::PortHandler::getPortHandler(dev);
    auto* bus = dynamixel::PacketHandler::getPacketHandler(2.0);
    if (!port->openPort() || !port->setBaudRate(115200)) return fprintf(stderr, "cannot open %s (led_monitor still running?)\n", dev), 1;

    int failed = 0;
    for (int id : IDS) {
        uint32_t pos = 0;
        bool ok = !on || (bus->read4ByteTxRx(port, id, ADDR_PRESENT, &pos) == COMM_SUCCESS &&
                          bus->write4ByteTxRx(port, id, ADDR_GOAL, pos) == COMM_SUCCESS);
        ok = ok && bus->write1ByteTxRx(port, id, ADDR_TORQUE, on) == COMM_SUCCESS;
        if (on && ok) printf("ID %d torque on, holding %.1f deg\n", id, int32_t(pos) * 360.0 / 4096);
        else printf("ID %d torque %s%s\n", id, argv[1], ok ? "" : "  FAILED");
        failed += !ok;
    }
    port->closePort();
    return failed != 0;
}
