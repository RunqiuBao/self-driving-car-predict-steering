### Information for setup

**to split a bag file**
`roscore`
`rosbag filter input.bag output.bag "t.secs <= 1284703931.86"`
*note:* The time here is a epoch UNIX time stamp (need conversion)
