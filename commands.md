python fk_check_distances.py --offline --tip-offset-cm 3


python fk_offline.py \
  --points fk_test_20261020_233810_points.csv --pairs fk_test_20261020_233810_pairs.csv \
  --urdf ~/Documents/so101/SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
  --tip-offset-cm 3 --offsets -4.09 109.14 -96.09 5.49 -0.13

python fk_offline.py \
  --points fk_test_20261020_232630_points.csv --pairs fk_test_20261020_232630_pairs.csv \
  --urdf ~/Documents/so101/SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
  --tip-offset-cm 3 --offsets -4.09 109.14 -96.09 5.49 -0.13 --exclude A


python fk_distance_test.py --port /dev/ttyACM0 --id so101_follower_vasco \
     --leader-port /dev/ttyACM1 --leader-id so101_leader_vasco \
     --urdf ~/Documents/so101/SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
     --points ZERO --tip-offset-cm 3


python ik_test.py --port /dev/ttyACM0 --id so101_follower_vasco \
  --leader-port /dev/ttyACM1 --leader-id so101_leader_vasco --tip-offset-cm 2.75

    
lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower_vasco --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=so101_leader_vasco
