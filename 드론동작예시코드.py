import time
from e_drone.drone import *
from e_drone.protocol import *

def eventAttitude(attitude):
    print("Yaw : ", str(attitude.yaw))

def count_down(count):
    for i in range(count, 0, -1):
        print("{0}".format(i))
        time.sleep(1)

drone = Drone()
drone.open()

current_time = 0.
start_time = time.time() 

drone.setEventHandler(DataType.Attitude, eventAttitude)

print("TakeOff")
drone.sendTakeOff()
count_down(3)

while True:
    if current_time < 5:
        command = 0
    elif current_time < 10:
        command = 15
    elif current_time < 15:
        command = 0
    else:
        print("-- Break!")
        break

    current_time = time.time() - start_time
    print(current_time, command)

    drone.sendControlPosition(0, 0, 0, 0, command, 5) #5초?
    time.sleep(0.1)

    drone.sendRequest(DeviceType.Drone, DataType.Attitude)
    time.sleep(0.1)

print("Return Home") #착륙
drone.sendFlightEvent(FlightEvent.Return)
count_down(3)

drone.close()
