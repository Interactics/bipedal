// Finds every Dynamixel on the bus: tries each common baud rate with Protocol 2.0 and 1.0.
//   run: ./build/dxl_scan [/dev/cu.usbserial-XXXX]      (about 1 min)
#include <cstdio>
#include <glob.h>
#include <vector>

#include "dynamixel_sdk.h"

int main(int argc, char** argv) {
    glob_t found{};
    glob("/dev/cu.usbserial-*", 0, nullptr, &found);
    const char* dev = argc > 1 ? argv[1] : found.gl_pathc ? found.gl_pathv[0] : "/dev/cu.usbserial-NOT-FOUND";
    auto* port = dynamixel::PortHandler::getPortHandler(dev);
    if (!port->openPort()) return fprintf(stderr, "cannot open %s (another program using it?)\n", dev), 1;
    printf("scanning %s\n", dev);

    int total = 0;
    for (int baud : {9600, 57600, 115200, 1000000, 2000000, 3000000, 4000000}) {
        if (!port->setBaudRate(baud)) { printf("%7d bps: not supported by the port\n", baud); continue; }
        auto* p2 = dynamixel::PacketHandler::getPacketHandler(2.0);
        std::vector<uint8_t> ids;
        p2->broadcastPing(port, ids);
        for (uint8_t id : ids) printf("%7d bps  protocol 2.0  ID %3d\n", baud, id), total++;

        auto* p1 = dynamixel::PacketHandler::getPacketHandler(1.0);  // no broadcast ping in 1.0
        for (int id = 0; id < 253; id++) {
            uint16_t model = 0;
            if (p1->ping(port, id, &model) == COMM_SUCCESS) printf("%7d bps  protocol 1.0  ID %3d  model %d\n", baud, id, model), total++;
        }
        printf("%7d bps done\n", baud);
        fflush(stdout);
    }
    printf("%d found%s\n", total, total ? "" : ": check servo power (battery on the hub) and the RS-485 cable on the U2D2's RS-485 port");
    port->closePort();
}
