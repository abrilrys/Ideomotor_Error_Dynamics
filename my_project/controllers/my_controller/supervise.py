from controller import Supervisor # type: ignore
import sys

TIME_STEP = 32

supervisor = Supervisor()

# do this once only
robot_node = supervisor.getFromDef("MY_ROBOT")

#
hand_node= supervisor.getFromDef("MY_ROBOT.HAND")

if robot_node is None:
    sys.stderr.write("No DEF MY_ROBOT node found in the current world file\n")
    sys.exit(1)
trans_field = robot_node.getField("translation")


if hand_node is None:
    sys.stderr.write("No DEF HAND node found in the current world file\n")
    sys.exit(1)
hand_trans_field = hand_node.getField("translation")

def getBodyTranslation():
    while supervisor.step(TIME_STEP) != -1:
        values = trans_field.getSFVec3f()
        print("MY_ROBOT is at position: %g %g %g" % (values[0], values[1], values[2]))
        break
    return values

def getHandTranslation():
    while supervisor.step(TIME_STEP) != -1:
        values_hand = trans_field.getSFVec3f()
        print("HAND is at position: %g %g %g" % (values_hand[0], values_hand[1], values_hand[2]))
        break
    return values_hand

# while supervisor.step(TIME_STEP) != -1:
#     # this is done repeatedly
#     values = trans_field.getSFVec3f()
#     values_hand=hand_trans_field.getSFVec3f()
#     #print("MY_ROBOT is at position: %g %g %g" % (values[0], values[1], values[2]))
#     #print("HAND is at position: %g %g %g" % (values_hand[0], values_hand[1], values_hand[2]))
    
    