# Fall-Pose-Algorithm

results folder contains all the detected falls, neither false or true.

sit_stand.pt is a yolov8 model trained for siting and standing activity. The datasetes are from https://universe.roboflow.com/apollo-solutions-dev/sitting-and-standing

keypoints_history contains the last 30 available keypoints extracted from the user. That will be process by any OpenAi for fall analysis.

keypoints are extracted from the yolov8 model the yolov8s.pt

After success detection of two fall after 2seconds the cv will capture the frame and run it inside predict_action.
Invalid Fall:
 Sitting
 Standing
else:
 VALID
