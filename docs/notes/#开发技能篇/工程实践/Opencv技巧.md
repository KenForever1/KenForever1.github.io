## Opencv 不能打开视频
Opencv处理video是通过后端处理的，常见的有ffmpeg、gstreamer。需要查看opencv在编译时，是否开启了对应功能。

```
// Open video
    cv::VideoCapture capture;
    std::cout << cv::getBuildInformation();
    std::string video_out_name = "output.mp4";

    std::cout << "oepn video file" << std::endl;
    capture.open(video_path.c_str());
    video_out_name =
        video_path.substr(video_path.find_last_of(OS_PATH_SEP) + 1);

    if (!capture.isOpened())
    {
        printf("can not open video : %s\n", video_path.c_str());
        return;
    }
```
如果openvideo失败，通过std::cout << cv::getBuildInformation();查看是否支持ffmpeg

```
Video I/O:
    DC1394:                      NO
    FFMPEG:                      NO
      avcodec:                   NO
      avformat:                  NO
      avutil:                    NO
      swscale:                   NO
      avresample:                NO
    GStreamer:                   NO
    v4l/v4l2:                    YES (linux/videodev2.h)
```
this means, you can use a webcam, but no video files at all.
you'll have to go all the way back and[build the opencv libs with either gstreamer or ffmpeg support](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)
参考：
[https://answers.opencv.org/question/193543/how-to-solve-problem-with-videocapture-loading-in-linux/](https://answers.opencv.org/question/193543/how-to-solve-problem-with-videocapture-loading-in-linux/)